// ============================================================================
// FlattenGlobalScalarReads.cpp  --  Rewrite read-only scalar-member reads of a
//                                    module-global struct into reads of a
//                                    synthetic per-member global, and record
//                                    the provenance in the flatten plan.
// ============================================================================

#include <string>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

#define DEBUG_TYPE "flatten-global-scalar-reads"

namespace hlfir_bridge {

namespace {

constexpr unsigned kTraceDepth = 16;

llvm::StringRef traceToGlobalSym(mlir::Value v) {
  for (unsigned d = 0; d < kTraceDepth && v; ++d) {
    mlir::Operation* def = v.getDefiningOp();
    if (!def) return {};
    if (auto addrOf = mlir::dyn_cast<fir::AddrOfOp>(def)) return addrOf.getSymbol().getRootReference().getValue();
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
      v = decl.getMemref();
      continue;
    }
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = conv.getValue();
      continue;
    }
    if (auto boxAddr = mlir::dyn_cast<fir::BoxAddrOp>(def)) {
      v = boxAddr.getVal();
      continue;
    }
    // A DESIGNATE IS NOT TRANSPARENT HERE -- DO NOT PEEL ONE.  For a module-level ARRAY of records
    // the component read does sit on an element select (`hlfir.designate %decl(%i)` -> ref<record>,
    // then `hlfir.designate %that{"x"}` -> ref<f64>), so peeling looks like the missing hop.  It is
    // not: the SCAN already peels that element select before calling this (the `elemDg` capture in
    // `run()`, which peels only a COMPONENT-LESS designate with indices).  Peeling unconditionally
    // here additionally collapses a NESTED component: for `eqn_idx%cont%end` the walk also visits
    // the inner `{"end"}` designate, and with a peel that designate resolves straight to the global
    // and is accepted with `member = "end"`, emitting `eqn_idx_end` where the rest of the pipeline
    // binds `eqn_idx_cont_end`.  The unbound leaf then bakes to 0 (`_bake_scalars` bakes every free
    // symbol it has no value for), and every access of the form `eqn_idx%beg + i - 1` reads `[-1]`:
    // `Memlet subset negative out-of-bounds`, in four kernel families at once.
    return {};
  }
  return {};
}

llvm::StringRef traceWriteToGlobalSym(mlir::Value v) {
  for (unsigned d = 0; d < kTraceDepth && v; ++d) {
    mlir::Operation* def = v.getDefiningOp();
    if (!def) return {};
    if (auto desig = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
      v = desig.getMemref();
      continue;
    }
    return traceToGlobalSym(v);
  }
  return {};
}

bool splitModuleScopeSymbol(llvm::StringRef sym, llvm::StringRef& mod, llvm::StringRef& entity) {
  llvm::StringRef s = sym;
  if (!s.consume_front("_QM")) return false;
  size_t e = llvm::StringRef::npos;
  for (size_t i = 0; i < s.size(); ++i)
    if (std::isupper(static_cast<unsigned char>(s[i]))) {
      if (s[i] != 'E') return false;
      e = i;
      break;
    }
  if (e == llvm::StringRef::npos || e == 0 || e + 1 >= s.size()) return false;
  mod = s.take_front(e);
  entity = s.drop_front(e + 1);
  for (char const c : entity)
    if (!std::islower(static_cast<unsigned char>(c)) && !std::isdigit(static_cast<unsigned char>(c)) && c != '_')
      return false;
  return true;
}

std::string scalarDtypeName(mlir::Type t) {
  if (t.isF32()) return "float32";
  if (t.isF64()) return "float64";
  if (t.isInteger(8)) return "int8";
  if (t.isInteger(16)) return "int16";
  if (t.isInteger(32)) return "int32";
  if (t.isInteger(64)) return "int64";
  if (t.isInteger(1) || mlir::isa<fir::LogicalType>(t)) return "bool";
  if (auto ct = mlir::dyn_cast<mlir::ComplexType>(t)) {
    mlir::Type const et = ct.getElementType();
    if (et.isF32()) return "complex64";
    if (et.isF64()) return "complex128";
  }
  return "";
}

bool isPlainComponentDesignate(hlfir::DesignateOp d) {
  return d.getComponent().has_value() && d.getIndices().empty() && d.getSubstring().empty() && !d.getComponentShape() &&
         !d.getShape() && d.getTypeparams().empty() && !d.getComplexPart().has_value() && d->getNumResults() == 1;
}

struct Candidate {
  hlfir::DesignateOp designate;
  std::string symbol;
  std::string module;
  std::string entity;
  std::string member;
  std::string dtype;
  mlir::Type scalarTy;
  /// For a module-level ARRAY of records: the element select this component read sits on, and the
  /// array's static shape.  Empty for a scalar struct global, which is the original (and still the
  /// only other) case.
  hlfir::DesignateOp elemDg;
  llvm::SmallVector<int64_t, 4> outerShape;
  /// RECORD-MAJOR: the entity is a static array of records whose fields all share the member's type,
  /// so ONE 1-D companion holds the whole record array and the member is selected by index
  /// arithmetic -- `arr(i)%x` -> `arr_cm(i * numFields + <x's ordinal>)`.  That is Fortran's layout
  /// exactly, so a caller passes one base pointer instead of gathering fifteen strided views.  MFC's
  /// own comment on `eos_coefficients` says the record exists to avoid those descriptors in the
  /// Riemann kernels, so the per-member form costs precisely what MFC designed against.
  unsigned memberIndex = 0;
  unsigned numFields = 0;
  bool recordMajor = false;
};

struct FlattenGlobalScalarReadsPass
    : public mlir::PassWrapper<FlattenGlobalScalarReadsPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenGlobalScalarReadsPass)

  llvm::StringRef getArgument() const final { return "hlfir-flatten-global-scalar-reads"; }
  llvm::StringRef getDescription() const final {
    return "Rewrite a never-written scalar component read of a module-scope "
           "fir.global record (``mod%entity%member``) into a read of a "
           "synthetic bodiless ``fir.global`` carrying that member alone, and "
           "record {symbol, module, entity, member, dtype} in the "
           "hlfir.flatten_plan side table so the bindings emitter can assign "
           "``symbol = entity%member`` at the call boundary.";
  }

  void runOnOperation() override {
    auto module = getOperation();

    llvm::StringSet<> writtenGlobals;
    module.walk([&](mlir::Operation* op) {
      mlir::Value target;
      if (auto store = mlir::dyn_cast<fir::StoreOp>(op))
        target = store.getMemref();
      else if (auto assign = mlir::dyn_cast<hlfir::AssignOp>(op))
        target = assign.getLhs();
      else
        return;
      llvm::StringRef const sym = traceWriteToGlobalSym(target);
      if (!sym.empty()) writtenGlobals.insert(sym);
    });

    llvm::StringSet<> takenNames;
    module.walk([&](fir::GlobalOp g) { takenNames.insert(g.getSymName()); });
    module.walk([&](hlfir::DeclareOp d) { takenNames.insert(d.getUniqName()); });

    llvm::SmallVector<Candidate, 4> candidates;
    llvm::StringSet<> rejected;
    module.walk([&](hlfir::DesignateOp d) {
      if (!isPlainComponentDesignate(d)) return;
      // The component read sits either directly on the global's declare (a scalar struct global)
      // or on an element select of it (a module-level ARRAY of records).
      hlfir::DesignateOp elemDg;
      mlir::Value base = d.getMemref();
      if (auto ed = mlir::dyn_cast_or_null<hlfir::DesignateOp>(base.getDefiningOp()))
        if (!ed.getComponent().has_value() && !ed.getIndices().empty()) {
          elemDg = ed;
          base = ed.getMemref();
        }
      llvm::StringRef const sym = traceToGlobalSym(base);
      if (sym.empty()) return;
      llvm::StringRef mod;
      llvm::StringRef entity;
      if (!splitModuleScopeSymbol(sym, mod, entity)) return;
      auto g = module.lookupSymbol<fir::GlobalOp>(sym);
      if (!g || (g.getConstant().has_value() && *g.getConstant())) return;

      // ACCEPT THE ARRAY-OF-RECORDS GLOBAL TOO.  A record-typed global is the original case; a
      // STATIC sequence of records is MFC's `type(eos_coefficients), dimension(num_fluids_max) ::
      // eos_coeffs`.  The shape must be static -- a dynamic one cannot name the companion's type
      // here, and bailing is the right answer rather than emitting something unresolvable.
      llvm::SmallVector<int64_t, 4> outerShape;
      mlir::Type recTy = g.getType();
      if (auto seq = mlir::dyn_cast<fir::SequenceType>(g.getType())) {
        if (seq.getDimension() != 1) return;
        for (int64_t d : seq.getShape())
          if (d == fir::SequenceType::getUnknownExtent()) return;
        outerShape.assign(seq.getShape().begin(), seq.getShape().end());
        recTy = seq.getEleTy();
      }
      if (!mlir::isa<fir::RecordType>(recTy)) return;
      if (!outerShape.empty() && !elemDg) return;   // array global must be read via an element select

      llvm::StringRef const member = d.getComponent()->getValue();
      auto refTy = mlir::dyn_cast<fir::ReferenceType>(d.getResult().getType());
      if (!refTy) return;
      std::string const dtype = scalarDtypeName(refTy.getEleTy());

      // One companion can hold the whole record array only when every field has the member's type --
      // MFC's `eos_coefficients` is fifteen `real(wp)` fields and qualifies.  Anything else keeps the
      // per-member form, which is the conservative answer rather than an error.  Restricted to a 1-D
      // record array because the flattened extent and the index arithmetic are written for that case.
      unsigned memberIndex = 0;
      unsigned numFields = 0;
      bool recordMajor = false;
      if (outerShape.size() == 1) {
        if (auto rec = mlir::dyn_cast<fir::RecordType>(recTy)) {
          memberIndex = rec.getFieldIndex(member);
          numFields = rec.getNumFields();
          recordMajor = true;
          for (auto const& f : rec.getTypeList())
            if (f.second != refTy.getEleTy()) recordMajor = false;
        }
      }
      std::string const newSym = recordMajor ? (llvm::Twine(sym) + "_cm").str()
                                             : (llvm::Twine(sym) + "_" + member).str();

      bool ok = !dtype.empty() && !writtenGlobals.contains(sym) && !takenNames.contains(newSym);
      for (mlir::Operation* user : d.getResult().getUsers())
        if (!mlir::isa<fir::LoadOp>(user)) ok = false;
      if (!ok) {
        rejected.insert(newSym);
        return;
      }
      Candidate cand{d, sym.str(), mod.str(), entity.str(), member.str(), dtype,
                     refTy.getEleTy(), elemDg, outerShape};
      cand.memberIndex = memberIndex;
      cand.numFields = numFields;
      cand.recordMajor = recordMajor;
      candidates.push_back(cand);
    });

    mlir::Builder b(&getContext());
    llvm::SmallVector<mlir::Attribute, 4> table;
    llvm::StringSet<> emitted;
    unsigned rewritten = 0;
    for (Candidate& c : candidates) {
      std::string const newSym = c.recordMajor ? (c.symbol + "_cm") : (c.symbol + "_" + c.member);
      if (rejected.contains(newSym)) continue;
      // THE COMPANION'S TYPE IS THE MEMBER'S OWN, EXTENDED BY THE RECORD ARRAY'S SHAPE.  For a
      // scalar struct global that is just the scalar; for a module-level array of records it is a
      // 1-D array of the scalar, indexed by the SAME record index the element select used -- so
      // `arr(i)%x` becomes `arr_x(i)` and the access is one level shallower, not a different shape.
      mlir::Type const compTy = c.scalarTy;
      llvm::SmallVector<int64_t, 4> compShape;
      if (c.recordMajor)
        compShape.push_back(static_cast<int64_t>(c.numFields) * c.outerShape.front());
      else
        compShape.append(c.outerShape.begin(), c.outerShape.end());
      mlir::Type const companionTy =
          compShape.empty() ? compTy : mlir::Type(fir::SequenceType::get(compShape, compTy));
      if (emitted.insert(newSym).second) {
        mlir::OpBuilder gb(&getContext());
        gb.setInsertionPointToEnd(module.getBody());
        gb.create<fir::GlobalOp>(c.designate.getLoc(), llvm::StringRef(newSym), companionTy,
                                 llvm::ArrayRef<mlir::NamedAttribute>{});
        llvm::SmallVector<mlir::NamedAttribute, 6> entry{
            b.getNamedAttr("symbol", b.getStringAttr(newSym)),
            b.getNamedAttr("module", b.getStringAttr(c.module)),
            b.getNamedAttr("entity", b.getStringAttr(c.entity)),
            b.getNamedAttr("member", b.getStringAttr(c.member)),
            b.getNamedAttr("dtype", b.getStringAttr(c.dtype)),
        };
        // Say so when the companion is an ARRAY.  A consumer that materialises the flat array needs
        // the record extent, and a consumer that does not can ignore the key -- which is why it is
        // additive rather than a change to the existing entry shape.
        if (!compShape.empty())
          entry.push_back(b.getNamedAttr("outer_shape", b.getI64ArrayAttr(compShape)));
        // A consumer MUST read these: record-major means one flattened array whose member is an index,
        // not a symbol per member.  Assuming the per-member shape would size the argument wrongly.
        if (c.recordMajor) {
          entry.push_back(b.getNamedAttr("record_major", b.getUnitAttr()));
          entry.push_back(b.getNamedAttr("field_count", b.getI64IntegerAttr(c.numFields)));
        }
        table.push_back(b.getDictionaryAttr(entry));
      }
      mlir::OpBuilder rb(c.designate);
      mlir::Location const loc = c.designate.getLoc();
      auto refTy = fir::ReferenceType::get(companionTy);
      auto addr = rb.create<fir::AddrOfOp>(loc, refTy, mlir::SymbolRefAttr::get(&getContext(), newSym));
      mlir::Value shape;
      if (!compShape.empty()) {
        llvm::SmallVector<mlir::Value, 4> dims;
        for (int64_t d : compShape)
          dims.push_back(rb.create<mlir::arith::ConstantIndexOp>(loc, d));
        shape = rb.create<fir::ShapeOp>(loc, dims).getResult();
      }
      auto decl = rb.create<hlfir::DeclareOp>(loc, addr.getResult(), newSym, shape);
      mlir::Value repl = decl.getResult(0);
      if (c.elemDg) {
        // `arr(i)%x` -> `arr_x(i)`: keep the element select, drop the component.  Record-major folds
        // the member's ordinal in ARITHMETICALLY -- `arr_cm(i * numFields + <ordinal>)` -- rather than
        // as a leading dimension, because a shape-carried ordinate is what the bridge mis-bounds.
        llvm::SmallVector<mlir::Value, 4> idx;
        if (c.recordMajor) {
          mlir::Value acc = rb.create<mlir::arith::ConstantIndexOp>(
              loc, static_cast<int64_t>(c.memberIndex));
          mlir::Value const nfields = rb.create<mlir::arith::ConstantIndexOp>(
              loc, static_cast<int64_t>(c.numFields));
          for (mlir::Value v : c.elemDg.getIndices()) {
            // The element select's index is NOT necessarily `index`-typed (the verifier rejected
            // `arith.muli` on a mixed pair), so normalise it before scaling.
            mlir::Value const vi = v.getType().isIndex()
                                       ? v
                                       : rb.create<mlir::arith::IndexCastOp>(loc, rb.getIndexType(), v);
            auto scaled = rb.create<mlir::arith::MulIOp>(loc, vi, nfields);
            acc = rb.create<mlir::arith::AddIOp>(loc, scaled, acc);
          }
          idx.push_back(acc);
        } else {
          for (mlir::Value v : c.elemDg.getIndices())
            idx.push_back(v);
        }
        repl = rb.create<hlfir::DesignateOp>(loc, fir::ReferenceType::get(compTy), decl.getResult(0),
                                             idx);
      }
      c.designate.getResult().replaceAllUsesWith(repl);
      c.designate.erase();
      ++rewritten;
    }

    if (table.empty()) return;

    llvm::SmallVector<mlir::NamedAttribute, 2> planAttrs;
    if (auto existing = module->getAttrOfType<mlir::DictionaryAttr>("hlfir.flatten_plan"))
      for (mlir::NamedAttribute na : existing)
        if (na.getName() != "synthetic_globals") planAttrs.push_back(na);
    planAttrs.push_back(b.getNamedAttr("synthetic_globals", b.getArrayAttr(table)));
    module->setAttr("hlfir.flatten_plan", b.getDictionaryAttr(planAttrs));

    LLVM_DEBUG(llvm::dbgs() << "FlattenGlobalScalarReads: " << table.size() << " synthetic global(s), " << rewritten
                            << " designate(s) rewritten\n");
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createFlattenGlobalScalarReadsPass() {
  return std::make_unique<FlattenGlobalScalarReadsPass>();
}

}  // namespace hlfir_bridge
