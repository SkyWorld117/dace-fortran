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
//     value).  For every named scalar global the remaining module only ever
//     reads, each ``fir.load`` of it becomes the constant; a ``fir.if`` whose
//     condition then folds to a constant is replaced by the branch it takes,
//     and a ``cf.cond_br`` (a lowered ``select case``) by a branch to its
//     target, the blocks no longer reached going with it.  The ``symbol-dce``
//     that follows drops the procedures only a dead branch called, so
//     ``hlfir-inline-all`` never sees them.
//
// Safety:
//     A global the remaining module stores to, assigns, passes to a call or
//     uses in any other way than a load is left alone (and reported): its
//     value may change within the entry.  The values must be those of every
//     run the result serves; a binding guards them at runtime
//     (dace_fortran.bindings.specialization).
// ============================================================================

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

/// Every load of ``global``'s address, through the declares and converts on the way; false (and no loads) if the
/// address has any other user, which may change the value.
bool onlyLoaded(fir::AddrOfOp addr, llvm::SmallVectorImpl<fir::LoadOp>& loads) {
  llvm::SmallVector<mlir::Value, 4> work{addr.getResult()};
  while (!work.empty()) {
    mlir::Value v = work.pop_back_val();
    for (mlir::OpOperand& use : v.getUses()) {
      mlir::Operation* user = use.getOwner();
      if (auto load = mlir::dyn_cast<fir::LoadOp>(user)) {
        loads.push_back(load);
      } else if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(user)) {
        for (mlir::Value r : decl->getResults()) work.push_back(r);
      } else if (auto conv = mlir::dyn_cast<fir::ConvertOp>(user)) {
        work.push_back(conv.getResult());
      } else {
        return false;
      }
    }
  }
  return true;
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

struct SpecializeGlobalsPass
    : public mlir::PassWrapper<SpecializeGlobalsPass, mlir::OperationPass<mlir::ModuleOp>> {
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
    auto attr = module->getAttrOfType<mlir::DictionaryAttr>("hlfir.specialized_globals");
    if (!attr || attr.empty()) return;
    llvm::StringMap<std::string> values;  // mangled symbol -> value text
    for (mlir::NamedAttribute entry : attr)
      if (auto text = mlir::dyn_cast<mlir::StringAttr>(entry.getValue()))
        values[globalSymbol(entry.getName().getValue())] = text.getValue().str();

    // Every address-of a named global, by symbol.
    llvm::StringMap<llvm::SmallVector<fir::AddrOfOp, 4>> addresses;
    module.walk([&](fir::AddrOfOp addr) {
      llvm::StringRef const sym = addr.getSymbol().getRootReference().getValue();
      if (values.count(sym)) addresses[sym].push_back(addr);
    });

    unsigned folded = 0;
    for (auto& [sym, text] : values) {
      auto it = addresses.find(sym);
      if (it == addresses.end()) continue;  // nothing reads it
      llvm::SmallVector<fir::LoadOp, 8> loads;
      bool readOnly = true;
      for (fir::AddrOfOp addr : it->second) readOnly = readOnly && onlyLoaded(addr, loads);
      if (!readOnly) {
        llvm::errs() << "hlfir-specialize-globals: left `" << sym << "` alone: the module writes or passes it on.\n";
        continue;
      }
      unsigned replaced = 0;
      for (fir::LoadOp load : loads) {
        mlir::OpBuilder b(load);
        mlir::Value constant = buildConstant(b, load.getLoc(), load.getType(), text);
        if (!constant) {
          llvm::errs() << "hlfir-specialize-globals: left `" << sym << "` alone: `" << text
                       << "` is no value of its type.\n";
          break;
        }
        load.replaceAllUsesWith(constant);
        load.erase();
        ++replaced;
      }
      llvm::errs() << "hlfir-specialize-globals: `" << sym << "` = " << text << " (" << replaced << " load(s)).\n";
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
    for (bool changed = true; changed;) {
      (void)mlir::applyPatternsGreedily(module, frozen);
      llvm::SmallVector<fir::IfOp, 16> ifs;
      module.walk([&](fir::IfOp op) { ifs.push_back(op); });
      changed = false;
      for (fir::IfOp op : llvm::reverse(ifs))
        if (foldConstantIf(op)) {
          changed = true;
          ++folded;
        }
    }
    llvm::errs() << "hlfir-specialize-globals: " << folded << " branch(es) folded.\n";
    llvm::errs().flush();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createSpecializeGlobalsPass() { return std::make_unique<SpecializeGlobalsPass>(); }

}  // namespace hlfir_bridge
