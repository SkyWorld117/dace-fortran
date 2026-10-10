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
//     variable, ``"module::name(k)%component"`` for one of element ``k`` of a
//     fixed-size array of records).  For every named scalar the remaining module only ever reads,
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
//     component of a module record or of an element of a module array of
//     records read at a constant subscript, the module reads and never writes --
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

/// The value of an integer subscript that is a literal, through ``fir.convert``s.
std::optional<int64_t> constantIndex(mlir::Value v) {
  for (int i = 0; i < 8 && v; ++i) {
    mlir::Operation* def = v.getDefiningOp();
    if (auto cst = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(def)) {
      if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue())) return attr.getInt();
      return std::nullopt;
    }
    auto conv = mlir::dyn_cast_or_null<fir::ConvertOp>(def);
    if (!conv) break;
    v = conv.getValue();
  }
  return std::nullopt;
}

/// ``array(k)`` with one scalar subscript: an element of a rank-1 array of records.
std::optional<int64_t> constantElement(hlfir::DesignateOp d) {
  if (d.getComponent().has_value() || d.getIndices().size() != 1 || !d.getSubstring().empty() ||
      d.getComplexPart().has_value())
    return std::nullopt;
  for (bool const triplet : d.getIsTriplet())
    if (triplet) return std::nullopt;
  return constantIndex(d.getIndices()[0]);
}

/// A scalar inside a module variable: the variable itself, its ``component`` (a record), or the ``component`` of its
/// element ``element`` (a fixed-size array of records, ``idwbuff(1)%beg``).
struct ScalarPath {
  std::optional<int64_t> element;
  std::string component;
};

/// Parses the part of a key after the variable's name: "", ``%component`` or ``(k)%component``; nullopt otherwise.
std::optional<ScalarPath> parseScalarPath(llvm::StringRef rest) {
  ScalarPath path;
  if (rest.consume_front("(")) {
    auto [number, after] = rest.split(')');
    int64_t k = 0;
    if (number.trim().getAsInteger(10, k) || after.size() == rest.size()) return std::nullopt;
    path.element = k;
    rest = after;
    if (!rest.starts_with("%")) return std::nullopt;  // an element of an array of records has no scalar value
  }
  if (rest.consume_front("%")) {
    if (rest.empty()) return std::nullopt;
    path.component = rest.lower();
  } else if (!rest.empty()) {
    return std::nullopt;
  }
  return path;
}

/// True when every user of ``v`` only reads through it: loads, and declares, converts and designators leading to
/// loads.
bool onlyReads(mlir::Value v, int depth = 0) {
  if (depth > 16) return false;
  for (mlir::OpOperand& use : v.getUses()) {
    mlir::Operation* user = use.getOwner();
    if (mlir::isa<fir::LoadOp>(user)) continue;
    if (mlir::isa<hlfir::DeclareOp, fir::ConvertOp>(user) ||
        (mlir::isa<hlfir::DesignateOp>(user) && mlir::cast<hlfir::DesignateOp>(user).getMemref() == v)) {
      for (mlir::Value r : user->getResults())
        if (!onlyReads(r, depth + 1)) return false;
      continue;
    }
    return false;
  }
  return true;
}

/// Every load of the scalar ``path`` names in ``global``, through the declares and converts on the way; false (and
/// no loads) if anything may change it: a user of its address other than a load, of its record or element other than
/// designators of other components, or a non-reading use of another element (whose subscript may be the same).
bool onlyLoaded(fir::AddrOfOp addr, llvm::SmallVectorImpl<fir::LoadOp>& loads, ScalarPath const& path = {}) {
  // How far a value has come along ``path``: at the variable, at its element, or at the scalar itself.
  enum Stage { Variable, Element, Target };
  auto next = [&](Stage stage) {
    if (stage == Variable && path.element) return Element;
    return Target;
  };
  Stage const first = (!path.element && path.component.empty()) ? Target : Variable;
  llvm::SmallVector<std::pair<mlir::Value, Stage>, 4> work{{addr.getResult(), first}};
  while (!work.empty()) {
    auto [v, stage] = work.pop_back_val();
    for (mlir::OpOperand& use : v.getUses()) {
      mlir::Operation* user = use.getOwner();
      auto designate = mlir::dyn_cast<hlfir::DesignateOp>(user);
      if (designate && designate.getMemref() != v) designate = nullptr;
      if (auto load = mlir::dyn_cast<fir::LoadOp>(user)) {
        if (stage == Target) loads.push_back(load);
      } else if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(user)) {
        for (mlir::Value r : decl->getResults()) work.push_back({r, stage});
      } else if (auto conv = mlir::dyn_cast<fir::ConvertOp>(user)) {
        work.push_back({conv.getResult(), stage});
      } else if (designate && stage == Variable && path.element) {
        if (constantElement(designate) == path.element)
          work.push_back({designate.getResult(), next(stage)});
        else if (!onlyReads(designate.getResult()))
          return false;
      } else if (designate && stage != Target && isScalarComponent(designate)) {
        if (*designate.getComponent() == path.component) work.push_back({designate.getResult(), Target});
      } else {
        return false;
      }
    }
  }
  return true;
}

/// The distinct literal subscripts ``global`` (a rank-1 array) is designated at, through declares and converts.
llvm::SmallVector<int64_t, 4> constantElements(llvm::ArrayRef<fir::AddrOfOp> addrs) {
  llvm::SmallVector<int64_t, 4> elements;
  llvm::SmallVector<mlir::Value, 8> work;
  for (fir::AddrOfOp addr : addrs) work.push_back(addr.getResult());
  while (!work.empty()) {
    mlir::Value v = work.pop_back_val();
    for (mlir::Operation* user : v.getUsers()) {
      if (mlir::isa<hlfir::DeclareOp, fir::ConvertOp>(user)) {
        for (mlir::Value r : user->getResults()) work.push_back(r);
      } else if (auto d = mlir::dyn_cast<hlfir::DesignateOp>(user)) {
        if (auto k = constantElement(d); k && !llvm::is_contained(elements, *k)) elements.push_back(*k);
      }
    }
  }
  llvm::sort(elements);
  return elements;
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
    // The variable itself when it is a scalar, each scalar component of a record, and each scalar component of every
    // element of a fixed-size rank-1 array of records read at a literal subscript.
    llvm::SmallVector<std::pair<ScalarPath, std::string>, 8> scalars;  // (path, Fortran type)
    mlir::Type type = global.getType();
    llvm::SmallVector<std::optional<int64_t>, 4> elements{std::nullopt};
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(type); seq && seq.getDimension() == 1 && seq.hasConstantShape()) {
      type = seq.getEleTy();
      elements.clear();
      for (int64_t k : constantElements(addrs)) elements.push_back(k);
    }
    if (auto record = mlir::dyn_cast<fir::RecordType>(type)) {
      for (std::optional<int64_t> element : elements)
        for (auto& [component, componentType] : record.getTypeList())
          if (std::string t = scalarTypeName(componentType); !t.empty())
            scalars.push_back({ScalarPath{element, component}, t});
    } else if (std::string t = scalarTypeName(type); !t.empty() && elements.front() == std::nullopt) {
      scalars.push_back({ScalarPath{}, t});
    }
    for (auto& [path, scalarType] : scalars) {
      llvm::SmallVector<fir::LoadOp, 8> loads;
      bool readOnly = true;
      for (fir::AddrOfOp addr : addrs) readOnly = readOnly && onlyLoaded(addr, loads, path);
      if (!readOnly || loads.empty()) continue;
      std::string key = mod + "::" + name;
      if (path.element) key += "(" + std::to_string(*path.element) + ")";
      if (!path.component.empty()) key += "%" + path.component;
      candidates.push_back(b.getNamedAttr(key, b.getStringAttr(scalarType)));
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
        // ``module::name`` or a mangled symbol, then the path to a scalar inside it: ``%component``, ``(k)%component``.
        llvm::StringRef const name = entry.getName().getValue();
        size_t const cut = name.find_first_of("(%");
        values[globalSymbol(name.take_front(cut)) + name.drop_front(std::min(cut, name.size())).lower()] =
            text.getValue().str();
      }

    // Every address-of a named global, by symbol.
    llvm::StringMap<llvm::SmallVector<fir::AddrOfOp, 4>> addresses;
    module.walk([&](fir::AddrOfOp addr) { addresses[addr.getSymbol().getRootReference().getValue()].push_back(addr); });

    unsigned folded = 0;
    for (auto& [key, text] : values) {
      size_t const cut = llvm::StringRef(key).find_first_of("(%");
      llvm::StringRef const sym = llvm::StringRef(key).take_front(cut);
      std::optional<ScalarPath> path = parseScalarPath(llvm::StringRef(key).drop_front(std::min(cut, key.size())));
      if (!path) {
        llvm::errs() << "hlfir-specialize-globals: left `" << key << "` alone: not a scalar of a module variable.\n";
        continue;
      }
      auto it = addresses.find(sym);
      if (it == addresses.end()) continue;  // nothing reads it
      llvm::SmallVector<fir::LoadOp, 8> loads;
      bool readOnly = true;
      for (fir::AddrOfOp addr : it->second) readOnly = readOnly && onlyLoaded(addr, loads, *path);
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
      // A parenthesised condition (``if ((a .and. b))``) reaches its branch through an ``hlfir.no_reassoc``, which
      // means nothing for a constant and would hide it from the folding.
      llvm::SmallVector<hlfir::NoReassocOp, 8> parens;
      module.walk([&](hlfir::NoReassocOp op) {
        if (op.getVal().getDefiningOp<mlir::arith::ConstantOp>()) parens.push_back(op);
      });
      for (hlfir::NoReassocOp op : parens) {
        op.replaceAllUsesWith(op.getVal());
        op.erase();
      }
      (void)mlir::applyPatternsGreedily(module, frozen, config);
      // Post-order, inner ifs first: folding one never touches an if still to come, while an outer if folded first
      // would erase the inner ones of the branch it drops.
      llvm::SmallVector<fir::IfOp, 16> ifs;
      module.walk<mlir::WalkOrder::PostOrder>([&](fir::IfOp op) { ifs.push_back(op); });
      changed = !parens.empty();
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
