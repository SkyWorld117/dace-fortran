// ============================================================================
// FlattenGlobalScalarReads.cpp  --  Rewrite read-only scalar-member reads of a
//                                    module-global struct into reads of a
//                                    synthetic per-member global, and record
//                                    the provenance in the flatten plan.
//
// The member may sit in a nested record (``entity%a%b`` lifts to
// ``entity_a_b``).  A module-global array of records with a static shape is
// handled the same way per element: ``entity(i)%member`` reads element ``i``
// of a synthetic global array ``entity_member`` of the array's shape and
// bounds, which the bindings fill from ``entity%member`` (the component over
// every element).
// ============================================================================

#include <string>

#include "bridge/trace_utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
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

bool isPlainComponentDesignate(hlfir::DesignateOp d) {
  return d.getComponent().has_value() && d.getIndices().empty() && d.getSubstring().empty() && !d.getComponentShape() &&
         !d.getShape() && d.getTypeparams().empty() && !d.getComplexPart().has_value() && d->getNumResults() == 1;
}

/// For ``entity(i, ...)%member``: the element designate ``entity(i, ...)`` (scalar subscripts only) the component
/// designate reads from; null when the component is taken off something else.
hlfir::DesignateOp elementDesignateOf(hlfir::DesignateOp component) {
  auto elem = component.getMemref().getDefiningOp<hlfir::DesignateOp>();
  if (!elem || elem.getComponent().has_value() || elem.getIndices().empty() || !elem.getSubstring().empty() ||
      elem.getComplexPart().has_value() || !elem.getTypeparams().empty())
    return {};
  for (bool const triplet : elem.getIsTriplet())
    if (triplet) return {};
  return elem;
}

struct Candidate {
  /// The leaf component designate, then the record components above it up to the root (``entity%a%b``: ``%b``, ``%a``).
  llvm::SmallVector<hlfir::DesignateOp, 2> chain;
  hlfir::DesignateOp designate;
  /// ``entity(i, ...)`` when ``entity`` is an array of records, else null.
  hlfir::DesignateOp element;
  /// The host array's declare (its shape operand gives the synthetic array's bounds); null for a scalar record.
  hlfir::DeclareOp hostDecl;
  std::string symbol;
  std::string module;
  std::string entity;
  std::string member;
  std::string dtype;
  mlir::Type scalarTy;
  /// Type of the synthetic global: ``scalarTy``, or an array of it in the host array's shape.
  mlir::Type globalTy;
  /// Name of the synthetic global.
  std::string newSym;
  /// RECORD-MAJOR (MFC's ``eos_coefficients``): one flattened companion ``<entity>_cm`` over a static 1-D
  /// array of records whose fields all share the member's type; ``arr(i)%x`` -> ``arr_cm(i*numFields + ordinal)``.
  /// That is Fortran's layout exactly, so a caller passes one base pointer instead of N strided views.
  bool recordMajor = false;
  unsigned memberIndex = 0;
  unsigned numFields = 0;
  int64_t outerExtent = 0;
};

struct FlattenGlobalScalarReadsPass
    : public mlir::PassWrapper<FlattenGlobalScalarReadsPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenGlobalScalarReadsPass)

  llvm::StringRef getArgument() const final { return "hlfir-flatten-global-scalar-reads"; }
  llvm::StringRef getDescription() const final {
    return "Rewrite a never-written scalar component read of a module-scope "
           "fir.global record (``mod%entity%member``, or ``entity(i)%member`` "
           "of a static array of records) into a read of a synthetic bodiless "
           "``fir.global`` carrying that member alone (an array of it for an "
           "array of records), and record {symbol, module, entity, member, "
           "dtype} in the hlfir.flatten_plan side table so the bindings "
           "emitter can assign ``symbol = entity%member`` at the call boundary.";
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
      // Walk up a nested record path ``entity%a%b`` to the component taken off the root.
      llvm::SmallVector<hlfir::DesignateOp, 2> chain{d};
      while (auto up = chain.back().getMemref().getDefiningOp<hlfir::DesignateOp>()) {
        if (!isPlainComponentDesignate(up)) break;
        chain.push_back(up);
      }
      hlfir::DesignateOp element = elementDesignateOf(chain.back());
      llvm::StringRef const sym = traceToGlobalSym(element ? element.getMemref() : chain.back().getMemref());
      if (sym.empty()) return;
      llvm::StringRef mod;
      llvm::StringRef entity;
      if (!splitModuleScopeSymbol(sym, mod, entity)) return;
      auto g = module.lookupSymbol<fir::GlobalOp>(sym);
      if (!g || (g.getConstant().has_value() && *g.getConstant())) return;
      // A scalar record, or an array of records with a static shape read element-wise through its declare.
      auto hostSeq = mlir::dyn_cast<fir::SequenceType>(g.getType());
      hlfir::DeclareOp hostDecl;
      if (element) {
        hostDecl = element.getMemref().getDefiningOp<hlfir::DeclareOp>();
        if (!hostSeq || hostSeq.hasDynamicExtents() || !hostDecl || !hostDecl.getShape() ||
            !mlir::isa<fir::RecordType>(hostSeq.getEleTy()))
          return;
      } else if (!mlir::isa<fir::RecordType>(g.getType())) {
        return;
      }

      // ``a%b`` for the table (the bindings read ``entity%a%b``), ``a_b`` for the synthetic symbol.
      std::string member;
      std::string symbolSuffix;
      for (hlfir::DesignateOp step : llvm::reverse(chain)) {
        llvm::StringRef const name = step.getComponent()->getValue();
        member += (member.empty() ? "" : "%") + name.str();
        symbolSuffix += (symbolSuffix.empty() ? "" : "_") + name.str();
      }
      auto refTy = mlir::dyn_cast<fir::ReferenceType>(d.getResult().getType());
      if (!refTy) return;
      std::string const dtype = dtypeName(refTy.getEleTy());
      mlir::Type const scalarTy = refTy.getEleTy();
      // RECORD-MAJOR: a static 1-D array of records whose fields all have the member's type.  One companion holds
      // the whole record array; the member is an INDEX (MFC's eos_coefficients is fifteen real fields).
      bool recordMajor = false;
      unsigned memberIndex = 0;
      unsigned numFields = 0;
      int64_t outerExtent = 0;
      if (element && chain.size() == 1 && hostSeq.getShape().size() == 1) {
        if (auto rec = mlir::dyn_cast<fir::RecordType>(hostSeq.getEleTy())) {
          memberIndex = rec.getFieldIndex(chain.front().getComponent()->getValue());
          numFields = rec.getNumFields();
          outerExtent = hostSeq.getShape().front();
          recordMajor = true;
          for (auto const& f : rec.getTypeList())
            if (f.second != scalarTy) recordMajor = false;
        }
      }
      std::string const newSym = recordMajor ? (llvm::Twine(sym) + "_cm").str()
                                             : (llvm::Twine(sym) + "_" + symbolSuffix).str();

      bool ok = !dtype.empty() && !writtenGlobals.contains(sym) && !takenNames.contains(newSym);
      for (mlir::Operation* user : d.getResult().getUsers())
        if (!mlir::isa<fir::LoadOp>(user)) ok = false;
      if (!ok) {
        rejected.insert(newSym);
        return;
      }
      mlir::Type const globalTy =
          recordMajor ? mlir::Type(fir::SequenceType::get({static_cast<int64_t>(numFields) * outerExtent}, scalarTy))
                      : (element ? mlir::Type(fir::SequenceType::get(hostSeq.getShape(), scalarTy)) : scalarTy);
      Candidate cand{chain, d, element, hostDecl, sym.str(), mod.str(), entity.str(), member, dtype, scalarTy, globalTy,
                     newSym};
      cand.recordMajor = recordMajor;
      cand.memberIndex = memberIndex;
      cand.numFields = numFields;
      cand.outerExtent = outerExtent;
      candidates.push_back(cand);
    });

    mlir::Builder b(&getContext());
    llvm::SmallVector<mlir::Attribute, 4> table;
    llvm::StringSet<> emitted;
    unsigned rewritten = 0;
    for (Candidate& c : candidates) {
      std::string const& newSym = c.newSym;
      if (rejected.contains(newSym)) continue;
      if (emitted.insert(newSym).second) {
        mlir::OpBuilder gb(&getContext());
        gb.setInsertionPointToEnd(module.getBody());
        gb.create<fir::GlobalOp>(c.designate.getLoc(), llvm::StringRef(newSym), c.globalTy,
                                 llvm::ArrayRef<mlir::NamedAttribute>{});
        llvm::SmallVector<mlir::NamedAttribute, 8> entry{
            b.getNamedAttr("symbol", b.getStringAttr(newSym)),
            b.getNamedAttr("module", b.getStringAttr(c.module)),
            b.getNamedAttr("entity", b.getStringAttr(c.entity)),
            b.getNamedAttr("member", b.getStringAttr(c.member)),
            b.getNamedAttr("dtype", b.getStringAttr(c.dtype)),
        };
        if (c.recordMajor) {
          // A consumer MUST read these: record-major means one flattened array whose member is an index.
          entry.push_back(b.getNamedAttr("outer_shape",
                                         b.getI64ArrayAttr({static_cast<int64_t>(c.numFields) * c.outerExtent})));
          entry.push_back(b.getNamedAttr("record_major", b.getUnitAttr()));
          entry.push_back(b.getNamedAttr("field_count", b.getI64IntegerAttr(c.numFields)));
        }
        table.push_back(b.getDictionaryAttr(entry));
      }
      mlir::OpBuilder rb(c.designate);
      mlir::Location const loc = c.designate.getLoc();
      auto addr = rb.create<fir::AddrOfOp>(loc, fir::ReferenceType::get(c.globalTy),
                                           mlir::SymbolRefAttr::get(&getContext(), newSym));
      mlir::Value shape = c.element ? c.hostDecl.getShape() : mlir::Value{};
      if (c.recordMajor) {
        llvm::SmallVector<mlir::Value, 1> dims;
        dims.push_back(rb.create<mlir::arith::ConstantIndexOp>(
            loc, static_cast<int64_t>(c.numFields) * c.outerExtent));
        shape = rb.create<fir::ShapeOp>(loc, dims).getResult();
      }
      auto decl = rb.create<hlfir::DeclareOp>(loc, addr.getResult(), newSym, shape);
      mlir::Value replacement = decl.getResult(0);
      if (c.element) {
        if (c.recordMajor) {
          // ``arr(i)%x`` -> ``arr_cm(i * numFields + <x's ordinal>)``: the ordinal is ARITHMETIC, not a leading
          // dimension (a shape-carried ordinate is what the bridge mis-bounds).
          llvm::SmallVector<mlir::Value, 1> idx;
          mlir::Value acc =
              rb.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(c.memberIndex));
          mlir::Value const nfields =
              rb.create<mlir::arith::ConstantIndexOp>(loc, static_cast<int64_t>(c.numFields));
          for (mlir::Value v : c.element.getIndices()) {
            mlir::Value const vi = v.getType().isIndex()
                                       ? v
                                       : rb.create<mlir::arith::IndexCastOp>(loc, rb.getIndexType(), v);
            acc = rb.create<mlir::arith::AddIOp>(loc, acc,
                                                 rb.create<mlir::arith::MulIOp>(loc, vi, nfields));
          }
          idx.push_back(acc);
          replacement =
              rb.create<hlfir::DesignateOp>(loc, fir::ReferenceType::get(c.scalarTy), replacement, idx).getResult();
        } else {
          // ``entity(i, ...)%member`` -> ``entity_member(i, ...)``, subscripted like the host array.
          replacement =
              rb.create<hlfir::DesignateOp>(loc, fir::ReferenceType::get(c.scalarTy), replacement, c.element.getIndices())
                  .getResult();
        }
      }
      c.designate.getResult().replaceAllUsesWith(replacement);
      // The leaf, then every component above it and the element designate that no other read still uses.
      for (hlfir::DesignateOp step : c.chain)
        if (step->use_empty()) step.erase();
      if (c.element && c.element->use_empty()) c.element.erase();
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
