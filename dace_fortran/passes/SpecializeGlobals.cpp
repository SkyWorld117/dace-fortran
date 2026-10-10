// ============================================================================
// SpecializeGlobals.cpp  --  Bake the values of named module variables into
// the code that reads them, and drop the branches that become dead, before
// anything is inlined.
// ============================================================================
// Problem:
//     A large entry (a solver's whole right-hand side) calls, behind runtime
//     configuration flags, every feature its code base has.  ``hlfir-inline-all``
//     inlines all of them, although a given configuration runs a handful: the
//     inlined module grows by orders of magnitude, the later passes slow to a
//     crawl, and features the bridge cannot lower (communication for physics
//     the configuration does not use, say) fail the pipeline.  Loops bounded by
//     a configuration variable stay non-constant for passes that need a
//     literal trip count.
//
// Fix:
//     The caller names module variables and their values (module attribute
//     ``hlfir.specialized_globals``: ``"module::name"`` or a mangled symbol ->
//     value; ``"module::name%component"`` for a scalar component of a record
//     variable).  For every named scalar the remaining module only ever reads,
//     each ``fir.load`` of it becomes the constant; a ``fir.if`` whose
//     condition then folds to a constant is replaced by the branch it takes,
//     and a ``cf.cond_br`` (a lowered ``select case``) by a branch to its
//     target, the blocks no longer reached going with it.  The ``symbol-dce``
//     that follows drops the procedures only a dead branch called, so
//     ``hlfir-inline-all`` never sees them.  Flang declares every variable of
//     a used module in each procedure; the declarations nothing reads any more
//     go too, or the bridge would still take those variables as inputs.
//
// Candidates:
//     Every non-constant integer, real or logical module scalar, and scalar
//     component of a module record, the module reads and never writes --
//     whatever the configuration of a run could fix -- is recorded first, before any value is applied, as the module
//     attribute ``hlfir.configuration_candidates`` (``"module::name"`` ->
//     Fortran type), so a caller can find out which values to fix.
//
// Safety:
//     A global the remaining module stores to, assigns, passes to a call or
//     uses in any other way than a load is left alone (and reported): its
//     value may change within the entry.  The values must be those of every
//     run the result serves; a binding guards them at runtime
//     (dace_fortran.bindings.specialization).
// ============================================================================

#include "bridge/extract_vars.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// The mangled symbol of a ``module::name`` key (``_QM<module>E<name>``), or the key itself when it is already one.
std::string globalSymbol(llvm::StringRef key) {
  auto [module, name] = key.split("::");
  if (name.empty()) return key.str();
  return ("_QM" + module.lower() + "E" + name.lower());
}

/// The constant ``text`` denotes for a scalar of type ``type``, or null if the type is not an integer, real or
/// logical, or the text does not parse.
mlir::Value buildConstant(mlir::OpBuilder& b, mlir::Location loc, mlir::Type type, llvm::StringRef text) {
  text = text.trim();
  if (auto logical = mlir::dyn_cast<fir::LogicalType>(type)) {
    std::string const t = text.lower();
    bool value = false;
    if (t == "1" || t == "true" || t == ".true.")
      value = true;
    else if (t != "0" && t != "false" && t != ".false.")
      return nullptr;
    auto bit = b.create<mlir::arith::ConstantOp>(loc, b.getBoolAttr(value));
    return b.create<fir::ConvertOp>(loc, logical, bit);
  }
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    int64_t value = 0;
    if (text.getAsInteger(10, value)) return nullptr;
    return b.create<mlir::arith::ConstantOp>(loc, b.getIntegerAttr(integer, value));
  }
  if (auto real = mlir::dyn_cast<mlir::FloatType>(type)) {
    llvm::APFloat value(real.getFloatSemantics());
    std::string normalised = text.str();
    for (char& c : normalised)
      if (c == 'd' || c == 'D') c = 'e';
    auto status = value.convertFromString(normalised, llvm::APFloat::rmNearestTiesToEven);
    if (!status) {
      llvm::consumeError(status.takeError());
      return nullptr;
    }
    return b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(real, value));
  }
  return nullptr;
}

/// ``global%component`` with no subscripts: a scalar component of a record taken whole.
bool isScalarComponent(hlfir::DesignateOp d) {
  return d.getComponent().has_value() && d.getIndices().empty() && d.getSubstring().empty() && !d.getComponentShape() &&
         !d.getShape() && d.getTypeparams().empty() && !d.getComplexPart().has_value();
}

/// Every load of ``global``'s address -- of its component ``component`` when one is given -- through the declares
/// and converts on the way; false (and no loads) if the address (the component) has any other user, which may change
/// the value. A load of the whole record and designators of its other components read or address other storage.
bool onlyLoaded(fir::AddrOfOp addr, llvm::SmallVectorImpl<fir::LoadOp>& loads, llvm::StringRef component = {}) {
  // (value, whether it addresses the variable itself -- the component, if one is given)
  llvm::SmallVector<std::pair<mlir::Value, bool>, 4> work{{addr.getResult(), component.empty()}};
  while (!work.empty()) {
    auto [v, target] = work.pop_back_val();
    for (mlir::OpOperand& use : v.getUses()) {
      mlir::Operation* user = use.getOwner();
      auto designate = mlir::dyn_cast<hlfir::DesignateOp>(user);
      if (auto load = mlir::dyn_cast<fir::LoadOp>(user)) {
        if (target) loads.push_back(load);
      } else if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(user)) {
        for (mlir::Value r : decl->getResults()) work.push_back({r, target});
      } else if (auto conv = mlir::dyn_cast<fir::ConvertOp>(user)) {
        work.push_back({conv.getResult(), target});
      } else if (!target && designate && designate.getMemref() == v && isScalarComponent(designate)) {
        if (*designate.getComponent() == component) work.push_back({designate.getResult(), true});
      } else {
        return false;
      }
    }
  }
  return true;
}

/// The Fortran type of an integer, real or logical scalar (``integer(4)``, ``real(8)``, ``logical(4)``), or "".
std::string scalarTypeName(mlir::Type type) {
  if (auto logical = mlir::dyn_cast<fir::LogicalType>(type))
    return "logical(" + std::to_string(logical.getFKind()) + ")";
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return "integer(" + std::to_string(integer.getWidth() / 8) + ")";
  if (auto real = mlir::dyn_cast<mlir::FloatType>(type)) return "real(" + std::to_string(real.getWidth() / 8) + ")";
  return "";
}

/// Record every non-constant integer, real or logical module scalar ``module`` reads and never writes as the module
/// attribute ``hlfir.configuration_candidates`` (``"module::name"`` -> Fortran type).
void recordCandidates(mlir::ModuleOp module) {
  llvm::StringMap<llvm::SmallVector<fir::AddrOfOp, 4>> addresses;
  module.walk([&](fir::AddrOfOp addr) { addresses[addr.getSymbol().getRootReference().getValue()].push_back(addr); });
  mlir::Builder b(module.getContext());
  llvm::SmallVector<mlir::NamedAttribute, 64> candidates;
  for (auto& [sym, addrs] : addresses) {
    auto global = module.lookupSymbol<fir::GlobalOp>(sym);
    if (!global || (global.getConstant().has_value() && *global.getConstant())) continue;
    auto [mod, name] = decodeModuleGlobalSymbol(sym.str());
    if (mod.empty()) continue;
    // The variable itself when it is a scalar, else each scalar component of a record.
    llvm::SmallVector<std::pair<std::string, std::string>, 8> scalars;  // (component or "", Fortran type)
    if (auto record = mlir::dyn_cast<fir::RecordType>(global.getType())) {
      for (auto& [component, type] : record.getTypeList())
        if (std::string t = scalarTypeName(type); !t.empty()) scalars.push_back({component, t});
    } else if (std::string t = scalarTypeName(global.getType()); !t.empty()) {
      scalars.push_back({"", t});
    }
    for (auto& [component, type] : scalars) {
      llvm::SmallVector<fir::LoadOp, 8> loads;
      bool readOnly = true;
      for (fir::AddrOfOp addr : addrs) readOnly = readOnly && onlyLoaded(addr, loads, component);
      if (!readOnly || loads.empty()) continue;
      std::string key = mod + "::" + name;
      if (!component.empty()) key += "%" + component;
      candidates.push_back(b.getNamedAttr(key, b.getStringAttr(type)));
    }
  }
  module->setAttr("hlfir.configuration_candidates", b.getDictionaryAttr(candidates));
}

/// The value of an ``i1`` condition that folds to a literal, through ``fir.convert``s.
std::optional<bool> constantCondition(mlir::Value v) {
  for (int i = 0; i < 16 && v; ++i) {
    mlir::Operation* def = v.getDefiningOp();
    if (auto cst = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(def)) {
      if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue())) return attr.getValue() != 0;
      if (auto attr = mlir::dyn_cast<mlir::BoolAttr>(cst.getValue())) return attr.getValue();
      return std::nullopt;
    }
    if (auto conv = mlir::dyn_cast_or_null<fir::ConvertOp>(def)) {
      v = conv.getValue();
      continue;
    }
    break;
  }
  return std::nullopt;
}

/// Replace a ``fir.if`` with a literal condition by the branch it takes; true if it did.
bool foldConstantIf(fir::IfOp ifOp) {
  std::optional<bool> cond = constantCondition(ifOp.getCondition());
  if (!cond) return false;
  mlir::Region& taken = *cond ? ifOp.getThenRegion() : ifOp.getElseRegion();
  if (taken.empty()) {
    // No else branch: an if without results simply goes away.
    if (ifOp.getNumResults() != 0) return false;
    ifOp.erase();
    return true;
  }
  mlir::Block& block = taken.front();
  auto result = mlir::cast<fir::ResultOp>(block.getTerminator());
  ifOp.replaceAllUsesWith(result.getOperands());
  result.erase();
  ifOp->getBlock()->getOperations().splice(ifOp->getIterator(), block.getOperations());
  ifOp.erase();
  return true;
}

struct SpecializeGlobalsPass : public mlir::PassWrapper<SpecializeGlobalsPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpecializeGlobalsPass)

  llvm::StringRef getArgument() const final { return "hlfir-specialize-globals"; }
  llvm::StringRef getDescription() const final {
    return "Replace loads of named, never-written module scalars by their given values and fold the branches "
           "that become constant, before inlining.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    recordCandidates(module);
    auto attr = module->getAttrOfType<mlir::DictionaryAttr>("hlfir.specialized_globals");
    if (!attr || attr.empty()) return;
    // mangled symbol, with ``%component`` for a component of a record -> value text
    llvm::StringMap<std::string> values;
    for (mlir::NamedAttribute entry : attr)
      if (auto text = mlir::dyn_cast<mlir::StringAttr>(entry.getValue())) {
        auto [variable, component] = entry.getName().getValue().split('%');
        std::string key = globalSymbol(variable);
        if (!component.empty()) key += "%" + component.lower();
        values[key] = text.getValue().str();
      }

    // Every address-of a named global, by symbol.
    llvm::StringMap<llvm::SmallVector<fir::AddrOfOp, 4>> addresses;
    module.walk([&](fir::AddrOfOp addr) { addresses[addr.getSymbol().getRootReference().getValue()].push_back(addr); });

    unsigned folded = 0;
    for (auto& [key, text] : values) {
      auto [sym, component] = llvm::StringRef(key).split('%');
      auto it = addresses.find(sym);
      if (it == addresses.end()) continue;  // nothing reads it
      llvm::SmallVector<fir::LoadOp, 8> loads;
      bool readOnly = true;
      for (fir::AddrOfOp addr : it->second) readOnly = readOnly && onlyLoaded(addr, loads, component);
      if (!readOnly) {
        llvm::errs() << "hlfir-specialize-globals: left `" << key << "` alone: the module writes or passes it on.\n";
        continue;
      }
      unsigned replaced = 0;
      for (fir::LoadOp load : loads) {
        mlir::OpBuilder b(load);
        mlir::Value constant = buildConstant(b, load.getLoc(), load.getType(), text);
        if (!constant) {
          llvm::errs() << "hlfir-specialize-globals: left `" << key << "` alone: `" << text
                       << "` is no value of its type.\n";
          break;
        }
        load.replaceAllUsesWith(constant);
        load.erase();
        ++replaced;
      }
      llvm::errs() << "hlfir-specialize-globals: `" << key << "` = " << text << " (" << replaced << " load(s)).\n";
    }

    // Fold what the constants decide: comparisons and logic first, then the branches on them, until nothing changes.
    // Only arith, cf and fir.convert patterns: enough to decide a condition and take the branch it decides, and nothing
    // the later passes do not expect.
    mlir::RewritePatternSet patterns(&getContext());
    for (mlir::RegisteredOperationName op : getContext().getRegisteredOperations()) {
      llvm::StringRef const dialect = op.getDialectNamespace();
      if (dialect == mlir::arith::ArithDialect::getDialectNamespace() ||
          dialect == mlir::cf::ControlFlowDialect::getDialectNamespace())
        op.getCanonicalizationPatterns(patterns, &getContext());
    }
    fir::ConvertOp::getCanonicalizationPatterns(patterns, &getContext());
    mlir::FrozenRewritePatternSet const frozen(std::move(patterns));
    // Normal region simplification erases the blocks a decided branch leaves unreachable. The default, aggressive level
    // also merges blocks that differ only in their operands -- the cases of a SELECT CASE that read different arrays --
    // into one block taking the array as a block argument, which the later passes cannot trace to its variable.
    mlir::GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Normal);
    for (bool changed = true; changed;) {
      (void)mlir::applyPatternsGreedily(module, frozen, config);
      // Post-order, inner ifs first: folding one never touches an if still to come, while an outer if folded first
      // would erase the inner ones of the branch it drops.
      llvm::SmallVector<fir::IfOp, 16> ifs;
      module.walk<mlir::WalkOrder::PostOrder>([&](fir::IfOp op) { ifs.push_back(op); });
      changed = false;
      for (fir::IfOp op : ifs)
        if (foldConstantIf(op)) {
          changed = true;
          ++folded;
        }
    }
    // Flang declares every variable of a used module in each procedure; the declares nothing reads any more -- the
    // specialised variables', and those only a dead branch used -- would still count as the entry's inputs.
    llvm::SmallVector<hlfir::DeclareOp, 16> unused;
    module.walk([&](hlfir::DeclareOp decl) {
      if (decl->use_empty() && mlir::isa_and_nonnull<fir::AddrOfOp>(decl.getMemref().getDefiningOp()))
        unused.push_back(decl);
    });
    for (hlfir::DeclareOp decl : unused) {
      llvm::SmallVector<mlir::Operation*, 4> feeding;
      for (mlir::Value operand : decl->getOperands())
        if (mlir::Operation* def = operand.getDefiningOp()) feeding.push_back(def);
      decl.erase();
      for (mlir::Operation* def : feeding)
        if (mlir::isOpTriviallyDead(def)) def->erase();
    }
    llvm::errs() << "hlfir-specialize-globals: " << folded << " branch(es) folded, " << unused.size()
                 << " unused module variable declaration(s) erased.\n";
    llvm::errs().flush();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createSpecializeGlobalsPass() { return std::make_unique<SpecializeGlobalsPass>(); }

}  // namespace hlfir_bridge
