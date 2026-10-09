// ============================================================================
// LiftAosPointerRecords.cpp  --  Lower AoS-of-pointer-records to flat
// per-member concatenation arrays.
// ============================================================================
//
// Motivating pattern (Graupel ``q``):
//
//     TYPE t_qx_ptr
//       REAL(wp), POINTER :: p(:), x(:,:)
//     END TYPE
//     TYPE(t_qx_ptr) :: q(N)
//
//     q(c1)%x => qv      ! N rebinds at function entry, ``c`` constant
//     q(c2)%x => qc
//     ...
//
//     ... = q(idx)%x(i, j)        ! ``idx`` may be a runtime value
//     q(idx)%x(i, j) = ...
//
// ``hlfir-flatten-structs`` rejects this shape (its docstring lists
// ``AoS-with-pointer-members`` as out of scope) and ``hlfir-rewrite-
// pointer-assigns`` rebinds only top-level pointer declares.  The
// result is the bridge emits ``q[(_i-1), (_j-1)]`` against a scalar-
// shape descriptor.
//
// Transformation
// --------------
// For each AoS-of-pointer-records ``q`` of static outer extent ``N``,
// per pointer-typed member ``m`` allocate a flat top-level transient
// ``q_<m>`` of shape ``(N, target_inner_shape...)``:
//
//   * Each rebind ``q(c)%m => target_c`` becomes an element-wise copy
//     loop into ``q_<m>(c, ...)`` at the rebind site.
//   * Each access ``q(idx)%m(i, j, ...)`` is rewritten to a direct
//     ``hlfir.designate`` over ``q_<m>(idx, i, j, ...)``.  A pointer member
//     takes over the lower bounds of the box it is assigned -- its target's
//     (``qa(0:n, -2:k)``) or a remap's (``q(c)%m(5:, 7:) => qa``) -- so the
//     inner dims of ``q_<m>`` carry them (a ``fir.shape_shift`` when one
//     differs from 1) and the subscripts index it unchanged.  The rebinds of
//     one member must agree on the lower bounds and static extents; runtime
//     extents are taken to agree.  A rebind to part of its target
//     (``=> qa(2:5, :)``) is not lifted.
//   * At the end of the rebind's block (before ``func.return`` for a
//     function-scope AoS) the matching copy-out loop writes ``q_<m>(c, ...)``
//     back to ``target_c`` so the original sees any in-body updates the alias
//     made.
//   * Between the rebind and that copy-out the concat is the live copy, so a
//     direct access to ``target_c`` there is redirected to it: an element
//     access ``target_c(i, j)`` becomes ``q_<m>(c, i, j)``; any other use
//     (whole-array, call) is bracketed by a copy-out before and a copy-in after.
//
// Pipeline placement: BEFORE ``hlfir-flatten-structs`` so flatten
// never sees the unsupported shape, and BEFORE ``hlfir-rewrite-
// pointer-assigns`` because the rebinds we eliminate here are not
// top-level pointer rebinds the sibling pass would handle anyway.
// ============================================================================

#include <optional>
#include <string>
#include <tuple>
#include <variant>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

struct MemberSpec {
  std::string name;
  fir::BoxType boxTy;
};

/// One rebind's view of a pointer member: its target's extents and the member's
/// lower bounds.  ``shape`` carries the per-dim static extent (from a
/// static-shape target's sequence type or a constant ``fir.shape`` /
/// ``fir.shape_shift`` operand); a dynamic dim holds
/// ``fir::SequenceType::getUnknownExtent()`` and takes its extent from
/// ``extentVals`` (an SSA value from the target declare's shape operand) or,
/// when that slot is null too, from ``boxSource``.
struct InnerShape {
  mlir::Type elemTy;
  llvm::SmallVector<int64_t, 4> shape;
  llvm::SmallVector<mlir::Value, 4> extentVals;
  /// Box value to query at use-site via ``fir.box_dims`` when no
  /// extent is statically known and no ``fir.shape`` op exposes the
  /// SSA extents.  Always null when one of the above already provides
  /// the per-dim extents.
  mlir::Value boxSource;
  /// Per-dim lower bound of the member: those of the box the rebind stores,
  /// i.e. the target's (``q(c)%x => qa`` with ``qa(0:n, -2:k)``) or a remap's
  /// (``q(c)%x(5:, 7:) => qa``).  ``lbounds[d]`` holds a static bound, or
  /// nullopt when ``lboundVals[d]`` carries it as an SSA value.  1 for every dim
  /// until ``setPointerLowerBounds`` has read the rebind.
  llvm::SmallVector<std::optional<int64_t>, 4> lbounds;
  llvm::SmallVector<mlir::Value, 4> lboundVals;

  /// Returns true when every dim has a static int extent (so the
  /// concat array can be allocated as a fully-typed static alloca).
  bool allStatic() const {
    for (auto d : shape)
      if (d == fir::SequenceType::getUnknownExtent()) return false;
    return !shape.empty();
  }

  /// True when every dim starts at 1, the case a plain ``fir.shape`` describes.
  bool unitLowerBounds() const {
    for (auto lb : lbounds)
      if (lb != 1) return false;
    return true;
  }
};

struct Candidate {
  hlfir::DeclareOp aosDecl;
  fir::SequenceType seqTy;
  fir::RecordType recordTy;
  int64_t N;
  /// The outer extent didn't fold to a compile-time constant (a DYNAMIC-extent
  /// AoS -- ICON ``recv_sp(nfields_sp)`` where ``nfields_sp`` is a runtime
  /// dummy).  ``N`` is left 0 by ``matchCandidate`` and filled in from the max
  /// 1-based rebind index in ``processCandidate`` (the concat only needs to
  /// cover the static ``recv1..recvM`` rebind slots the access loop indexes).
  bool dynamicExtent = false;
  llvm::SmallVector<MemberSpec, 4> members;
};

struct Rebind {
  int64_t outerIdx;  ///< 1-based, captured raw
  std::string memberName;
  hlfir::DeclareOp targetDecl;
  fir::StoreOp store;
  mlir::Value boxValue;  ///< the pointer box assigned by the rebind (the view's target box)
};

std::optional<int64_t> constInt(mlir::Value v) {
  if (!v) return std::nullopt;
  auto* def = v.getDefiningOp();
  if (!def) return std::nullopt;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue())) return ia.getInt();
  return std::nullopt;
}

/// ``constInt`` that ALSO forwards through the storage-transparent + clamp +
/// store-then-load shapes Flang emits for a loop counter or a dummy bound to a
/// literal: ``fir.convert``, the ``max(x, 0)`` automatic-array clamp
/// (``arith.select`` with a constant-0 arm), and ``fir.load %ref`` resolved to
/// the nearest preceding ``fir.store`` / ``hlfir.assign`` of a constant in the
/// SAME block (peeling ``hlfir.declare`` aliases of the storage).  ICON's
/// ``mult_mixprec`` builds ``recv_dp(nfields_dp)`` and rebinds
/// ``recv_dp(icount)%p`` via an incremented ``icount`` counter -- both the AoS
/// extent and the per-rebind index are store-then-load literals that the
/// arith.constant-only ``constInt`` misses (nothing in the pipeline folds them
/// because the counter/dummy stores are never store-to-load forwarded).  The
/// rebind setup is straight-line, so same-block nearest-store resolution is
/// sufficient; cross-block shapes fall back to ``nullopt`` (candidate skipped).
std::optional<int64_t> constIntForwarded(mlir::Value v) {
  auto isZeroConst = [](mlir::Value x) -> bool {
    if (auto c = x.getDefiningOp<mlir::arith::ConstantOp>())
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) return ia.getInt() == 0;
    return false;
  };
  for (int hop = 0; hop < 64 && v; ++hop) {
    auto* def = v.getDefiningOp();
    if (!def) return std::nullopt;
    if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) return ia.getInt();
      return std::nullopt;
    }
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = cv.getValue();
      continue;
    }
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
      // ``max(x, 0)`` automatic-array clamp: one arm is the constant 0; follow
      // the other.  A genuine non-zero ``select`` is not a known constant.
      if (isZeroConst(sel.getFalseValue())) {
        v = sel.getTrueValue();
        continue;
      }
      if (isZeroConst(sel.getTrueValue())) {
        v = sel.getFalseValue();
        continue;
      }
      return std::nullopt;
    }
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
      // The load's storage and every ``hlfir.declare`` aliasing it (a store may
      // target the declare RESULT or its underlying memref -- both the same
      // storage).
      llvm::SmallVector<mlir::Value, 4> refs;
      for (mlir::Value r = ld.getMemref(); r;) {
        refs.push_back(r);
        auto dc = r.getDefiningOp<hlfir::DeclareOp>();
        if (!dc || refs.size() >= 4) break;
        r = dc.getMemref();
      }
      mlir::Operation* best = nullptr;
      mlir::Value bestVal;
      auto consider = [&](mlir::Operation* op, mlir::Value target, mlir::Value value) {
        if (op->getBlock() != ld->getBlock() || !op->isBeforeInBlock(ld)) return;
        bool match = false;
        for (auto rr : refs)
          if (rr == target) match = true;
        if (!match) return;
        if (!best || best->isBeforeInBlock(op)) {
          best = op;
          bestVal = value;
        }
      };
      for (auto rr : refs)
        for (auto* u : rr.getUsers()) {
          if (auto st = mlir::dyn_cast<fir::StoreOp>(u))
            consider(st, st.getMemref(), st.getValue());
          else if (auto as = mlir::dyn_cast<hlfir::AssignOp>(u))
            consider(as, as.getOperand(1), as.getOperand(0));
        }
      if (!best) return std::nullopt;
      v = bestVal;
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

/// ``constInt`` through ``fir.convert`` only: Flang spells a literal lower bound as an integer constant converted to
/// index.  Unlike ``constIntForwarded`` it does not guess through clamps or stores, which would be unsound for a bound.
std::optional<int64_t> constIntThroughConvert(mlir::Value v) {
  while (auto cv = v ? v.getDefiningOp<fir::ConvertOp>() : fir::ConvertOp{}) v = cv.getValue();
  return constInt(v);
}

/// The origins (per-dim lower bounds) a ``fir.shape_shift`` / ``fir.shift`` carries; empty for a plain ``fir.shape``,
/// no shape at all (every dim starts at 1) or any other shape operand.
llvm::SmallVector<mlir::Value, 4> shapeOrigins(mlir::Value shape) {
  if (auto shapeShift = shape ? shape.getDefiningOp<fir::ShapeShiftOp>() : fir::ShapeShiftOp{}) {
    auto origins = shapeShift.getOrigins();  // a fresh vector per call
    return {origins.begin(), origins.end()};
  }
  if (auto shift = shape ? shape.getDefiningOp<fir::ShiftOp>() : fir::ShiftOp{}) {
    auto origins = shift.getOrigins();
    return {origins.begin(), origins.end()};
  }
  return {};
}

/// The extents a ``fir.shape`` / ``fir.shape_shift`` carries; empty for a ``fir.shift`` (the extents then come from a
/// descriptor), no shape at all or any other shape operand.
llvm::SmallVector<mlir::Value, 4> shapeExtents(mlir::Value shape) {
  if (auto shapeOp = shape ? shape.getDefiningOp<fir::ShapeOp>() : fir::ShapeOp{}) {
    auto extents = shapeOp.getExtents();
    return {extents.begin(), extents.end()};
  }
  if (auto shapeShift = shape ? shape.getDefiningOp<fir::ShapeShiftOp>() : fir::ShapeShiftOp{}) {
    auto extents = shapeShift.getExtents();  // a fresh vector per call
    return {extents.begin(), extents.end()};
  }
  return {};
}

/// True when ``shape`` starts every dim at 1: none, or a plain ``fir.shape``.
bool isUnitShape(mlir::Value shape) { return !shape || mlir::isa_and_nonnull<fir::ShapeOp>(shape.getDefiningOp()); }

/// True when two index-like values provably hold the same number: the same value, equal constants, or the same
/// side-effect-free computation (``arith``, ``fir.convert``, ``fir.box_dims``) over such values.  A load of an
/// ``intent(in)`` dummy argument stands for the dummy, which the procedure cannot change; Flang computes each
/// explicit-shape dummy's bounds from its own loads of the bound variables.
bool sameIndexValue(mlir::Value a, mlir::Value b, unsigned depth = 0) {
  if (a == b) return true;
  if (!a || !b || a.getType() != b.getType() || depth > 8) return false;
  auto ca = constIntThroughConvert(a);
  auto cb = constIntThroughConvert(b);
  if (ca || cb) return ca == cb;
  auto* da = a.getDefiningOp();
  auto* db = b.getDefiningOp();
  if (!da || !db || da->getName() != db->getName() || da->getAttrDictionary() != db->getAttrDictionary() ||
      mlir::cast<mlir::OpResult>(a).getResultNumber() != mlir::cast<mlir::OpResult>(b).getResultNumber())
    return false;
  if (auto la = mlir::dyn_cast<fir::LoadOp>(da)) {
    auto decl = la.getMemref().getDefiningOp<hlfir::DeclareOp>();
    auto attrs = decl ? decl.getFortranAttrs() : std::nullopt;
    return la.getMemref() == mlir::cast<fir::LoadOp>(db).getMemref() && attrs &&
           bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::intent_in);
  }
  if (!mlir::isa<mlir::arith::ArithDialect>(da->getDialect()) && !mlir::isa<fir::ConvertOp, fir::BoxDimsOp>(da))
    return false;
  for (auto [oa, ob] : llvm::zip_equal(da->getOperands(), db->getOperands()))
    if (!sameIndexValue(oa, ob, depth + 1)) return false;
  return true;
}

/// True when ``v`` provably is the extent of ``decl``'s dim ``d``: the extent its shape operand gives, or a
/// ``fir.box_dims`` query of that dim.
bool isExtentOf(mlir::Value v, hlfir::DeclareOp decl, unsigned d) {
  auto extents = shapeExtents(decl.getShape());
  if (d < extents.size() && sameIndexValue(v, extents[d])) return true;
  auto boxDims = v.getDefiningOp<fir::BoxDimsOp>();
  return boxDims && mlir::cast<mlir::OpResult>(v).getResultNumber() == 1 &&
         llvm::is_contained(decl->getResults(), boxDims.getVal()) && constIntThroughConvert(boxDims.getDim()) == d;
}

/// True when ``ub`` provably is the upper bound of ``decl``'s dim ``d``, given its lower bound ``lb`` (null: 1):
/// ``lb + extent - 1`` as Flang spells it, or the extent itself when ``lb`` is 1.
bool isUpperBoundOf(mlir::Value ub, hlfir::DeclareOp decl, unsigned d, mlir::Value lb) {
  auto isLowerBound = [&](mlir::Value v) { return lb ? sameIndexValue(v, lb) : constIntThroughConvert(v) == 1; };
  if ((!lb || constIntThroughConvert(lb) == 1) && isExtentOf(ub, decl, d)) return true;
  auto sub = ub.getDefiningOp<mlir::arith::SubIOp>();
  auto add = sub ? sub.getLhs().getDefiningOp<mlir::arith::AddIOp>() : mlir::arith::AddIOp{};
  if (!add || constIntThroughConvert(sub.getRhs()) != 1) return false;
  return (isLowerBound(add.getLhs()) && isExtentOf(add.getRhs(), decl, d)) ||
         (isLowerBound(add.getRhs()) && isExtentOf(add.getLhs(), decl, d));
}

/// True when ``dg`` designates the whole of the array its memref declares, as Flang spells ``t(:, :)``: one
/// ``lb:ub:1`` triplet per dim over the declare's own bounds.  Like any section it starts every dim at 1.
bool isWholeSection(hlfir::DesignateOp dg) {
  auto decl = dg.getMemref().getDefiningOp<hlfir::DeclareOp>();
  if (!decl || dg.getComponent().has_value() || !dg.getSubstring().empty() || dg.getComplexPart().has_value() ||
      !dg.getTypeparams().empty())
    return false;
  auto isTriplet = dg.getIsTriplet();
  auto indices = dg.getIndices();
  auto origins = shapeOrigins(decl.getShape());
  if (indices.size() != 3 * isTriplet.size() || (!origins.empty() && origins.size() != isTriplet.size())) return false;
  for (unsigned d = 0; d < isTriplet.size(); ++d) {
    mlir::Value const lb = origins.empty() ? mlir::Value{} : origins[d];
    mlir::Value const lo = indices[3 * d];
    if (!isTriplet[d] || !(lb ? sameIndexValue(lo, lb) : constIntThroughConvert(lo) == 1) ||
        !isUpperBoundOf(indices[(3 * d) + 1], decl, d, lb) || constIntThroughConvert(indices[(3 * d) + 2]) != 1)
      return false;
  }
  return true;
}

/// True when ``shape``, the shape operand of the embox or rebox that builds a rebind's box over ``decl``, keeps
/// ``decl``'s ``rank`` extents (no rank or extent remap), so the box sees the elements where ``decl`` does.  A
/// ``fir.shift`` or no shape keeps them by construction.
bool keepsExtents(mlir::Value shape, hlfir::DeclareOp decl, size_t rank) {
  if (!shape || shape.getDefiningOp<fir::ShiftOp>()) return true;
  auto extents = shapeExtents(shape);
  if (extents.size() != rank) return false;
  for (unsigned d = 0; d < rank; ++d)
    if (!isExtentOf(extents[d], decl, d)) return false;
  return true;
}

/// Element type stripped of the box-of-pointer wrappers (the pointer's
/// shape itself is assumed-shape ``?x?...`` for a generic
/// ``REAL, POINTER :: x(:,:)`` declaration, so the EXTENTS come from a
/// rebind target instead -- see ``innerShapeFromTargetDecl``).
mlir::Type elemTypeOfPtrMember(fir::BoxType boxTy) {
  auto inner = boxTy.getEleTy();
  if (auto p = mlir::dyn_cast<fir::PointerType>(inner)) inner = p.getEleTy();
  if (auto h = mlir::dyn_cast<fir::HeapType>(inner)) inner = h.getEleTy();
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(inner)) return seq.getEleTy();
  return inner;
}

/// Placeholder inner shape from the member box TYPE alone, for a matched AoS
/// with NO rebinds (the dead ``recv_sp`` exchange scaffolding when
/// ``nfields_sp`` folds to 0).  With no rebind target there is no source for
/// the assumed-shape ``?x?...`` extents, but every access to such an AoS is in
/// a 0-trip exchange loop (dead, never executed), so a size-1 record per inner
/// dim is a safe, fully-static placeholder: it keeps the concat transient
/// static (no dynamic extents to resolve), registers the ``<name>_p``
/// descriptor the dead accesses need, and is never touched at runtime.
std::optional<InnerShape> innerShapeFromMemberBox(fir::BoxType boxTy) {
  auto inner = boxTy.getEleTy();
  if (auto p = mlir::dyn_cast<fir::PointerType>(inner)) inner = p.getEleTy();
  if (auto h = mlir::dyn_cast<fir::HeapType>(inner)) inner = h.getEleTy();
  auto seq = mlir::dyn_cast<fir::SequenceType>(inner);
  if (!seq) return std::nullopt;
  InnerShape s;
  s.elemTy = seq.getEleTy();
  unsigned const rank = seq.getShape().size();
  s.shape.assign(rank, 1);
  s.extentVals.assign(rank, mlir::Value{});
  s.lbounds.assign(rank, 1);
  s.lboundVals.assign(rank, mlir::Value{});
  return s;
}

/// Element type and per-dim extents of a rebind target's declare (e.g.
/// ``qa(n, k)``), from its ``fir.shape`` / ``fir.shape_shift`` operand or, for a
/// static-shape local, its sequence type.  ``shape`` slots with
/// ``getUnknownExtent`` are dynamic; ``extentVals`` slots with a non-null Value
/// carry an SSA extent that already dominates the declare.  Slots where the dim
/// is dynamic AND no SSA extent is available (an assumed-shape target) leave
/// ``extentVals[d]`` null -- the caller emits a ``fir.box_dims`` on
/// ``boxSource`` to recover it.  The lower bounds stay 1 here; they are the
/// member's, read from the rebind (``setPointerLowerBounds``).
std::optional<InnerShape> innerShapeFromTargetDecl(hlfir::DeclareOp targetDecl) {
  mlir::Type eleTy;
  unsigned rank = 0;
  // Find the element type + rank from whichever typed view of the
  // declare carries them (the memref or the box-typed result).
  auto memrefTy = targetDecl.getMemref().getType();
  if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(memrefTy)) {
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(refTy.getEleTy())) {
      eleTy = seq.getEleTy();
      rank = seq.getShape().size();
    }
  } else if (auto boxTy = mlir::dyn_cast<fir::BoxType>(memrefTy)) {
    auto inner = boxTy.getEleTy();
    if (auto p = mlir::dyn_cast<fir::PointerType>(inner)) inner = p.getEleTy();
    if (auto h = mlir::dyn_cast<fir::HeapType>(inner)) inner = h.getEleTy();
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(inner)) {
      eleTy = seq.getEleTy();
      rank = seq.getShape().size();
    }
  }
  if (!eleTy || rank == 0) return std::nullopt;

  InnerShape s;
  s.elemTy = eleTy;
  s.lbounds.assign(rank, 1);
  s.lboundVals.assign(rank, mlir::Value{});
  auto addExtent = [&s](mlir::Value ext) {
    if (auto c = constInt(ext)) {
      s.shape.push_back(*c);
      s.extentVals.push_back(mlir::Value{});
    } else {
      s.shape.push_back(fir::SequenceType::getUnknownExtent());
      s.extentVals.push_back(ext);
    }
  };

  // First try the declare's ``shape`` operand: when present it gives
  // ready-to-use SSA per-dim extents.
  if (auto shapeOper = targetDecl.getShape()) {
    auto* shapeDef = shapeOper.getDefiningOp();
    if (auto shapeOp = mlir::dyn_cast_or_null<fir::ShapeOp>(shapeDef)) {
      for (auto ext : shapeOp.getExtents()) addExtent(ext);
      return s;
    }
    // ``qa(0:n, -2:k)``: lower bound / extent pairs.
    if (auto shapeShift = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shapeDef)) {
      auto extents = shapeShift.getExtents();
      if (extents.size() != rank) return std::nullopt;
      for (auto ext : extents) addExtent(ext);
      return s;
    }
  }

  // Fallback: pure-static seq type, no shape operand needed.
  if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(memrefTy)) {
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(refTy.getEleTy())) {
      bool allStatic = true;
      for (auto d : seq.getShape()) {
        if (d == fir::SequenceType::getUnknownExtent()) {
          allStatic = false;
          break;
        }
      }
      if (allStatic) {
        for (auto d : seq.getShape()) {
          s.shape.push_back(d);
          s.extentVals.push_back(mlir::Value{});
        }
        return s;
      }
    }
  }

  // Last resort: assumed-shape box-of-array dummy.  Record the box
  // value as the "source" so ``emitBoxDims`` can extract extents at
  // the insertion point.
  s.shape.assign(rank, fir::SequenceType::getUnknownExtent());
  s.extentVals.assign(rank, mlir::Value{});
  s.boxSource = targetDecl.getResult(0);
  return s;
}

std::optional<Candidate> matchCandidate(hlfir::DeclareOp d) {
  auto refTy = mlir::dyn_cast<fir::ReferenceType>(d.getMemref().getType());
  if (!refTy) return std::nullopt;
  auto seqTy = mlir::dyn_cast<fir::SequenceType>(refTy.getEleTy());
  if (!seqTy) return std::nullopt;
  if (seqTy.getShape().size() != 1) return std::nullopt;
  // The concat's outer dim takes the AoS subscripts unchanged and starts at 1, so the AoS must start at 1 too.
  for (auto origin : shapeOrigins(d.getShape()))
    if (constIntThroughConvert(origin) != 1) return std::nullopt;
  auto N = seqTy.getShape()[0];
  if (N == fir::SequenceType::getUnknownExtent()) {
    // Dynamic-shaped AoS whose outer EXTENT OPERAND is a compile-time constant
    // (ICON ``recv_dp(nfields_dp)`` -- ``nfields_dp`` folds to a literal 2/3
    // after monomorphisation but the alloca stays ``array<?>``).  Recover the
    // constant from the declare's shape operand so the flat ``(N, inner)``
    // transient can still be allocated with a static outer dim.
    if (auto shapeVal = d.getShape())
      if (auto shapeOp = shapeVal.getDefiningOp<fir::ShapeOp>())
        if (shapeOp.getExtents().size() == 1)
          if (auto c = constIntForwarded(shapeOp.getExtents()[0])) N = *c;
  }
  // ``N`` still UNKNOWN after the constant-forward attempt is a DYNAMIC-extent
  // AoS: ICON ``recv_sp(nfields_sp)`` where ``nfields_sp`` is a runtime dummy
  // that never folds (unlike ``recv_dp``'s ``nfields_dp`` in the ``seq``
  // variants).  Don't skip it -- its ``recv_sp(k)%p`` accesses (a
  // ``DO k = 1, nfields_sp`` exchange loop) still emit ``<name>_p`` SDFG access
  // nodes, so the flat transient MUST be registered or ``prune_unused_arrays``
  // hits a KeyError on the dangling access.  The rebinds are the STATIC
  // ``recv1_sp .. recvM_sp`` slots and the loop index is bounded by
  // ``nfields_sp <= M``, so ``processCandidate`` fills ``N`` with the max
  // 1-based rebind index -- a static ``(M, inner)`` concat that covers every
  // access.  A genuinely EMPTY static AoS (``N == 0``) gets a size-1 placeholder
  // in ``createConcatStorage`` (all uses dead).  Only a negative extent is
  // unliftable.
  bool const dynamicExtent = (N == fir::SequenceType::getUnknownExtent());
  if (!dynamicExtent && N < 0) return std::nullopt;
  auto recordTy = mlir::dyn_cast<fir::RecordType>(seqTy.getEleTy());
  if (!recordTy) return std::nullopt;
  Candidate c;
  c.aosDecl = d;
  c.seqTy = seqTy;
  c.recordTy = recordTy;
  c.N = dynamicExtent ? 0 : N;  // dynamic: filled from the max rebind index in processCandidate
  c.dynamicExtent = dynamicExtent;
  for (auto& member : recordTy.getTypeList()) {
    auto boxTy = mlir::dyn_cast<fir::BoxType>(member.second);
    if (!boxTy) return std::nullopt;
    auto inner = boxTy.getEleTy();
    // Only POINTER members (``fir.ptr``) belong to this pass -- it lifts the
    // ``q(c)%m => target`` rebind pattern.  An ALLOCATABLE member
    // (``fir.heap``) is never rebound with ``=>``; it is allocated with
    // ``allocate(A(i)%m(...))`` and belongs to flatten-structs' Phase 5c-A
    // (padded AoS companion) / ``hlfir-lift-alloc-array-of-records``.  Matching
    // a heap member here mints a bogus size-1 placeholder companion via the
    // ``rebinds.empty()`` "dead exchange" path and silently steals the live
    // member reads, so leave it for the alloc path.
    if (!mlir::isa<fir::PointerType>(inner)) return std::nullopt;
    c.members.push_back({member.first, boxTy});
  }
  if (c.members.empty()) return std::nullopt;
  return c;
}

hlfir::DeclareOp traceTarget(mlir::Value v) {
  for (int hop = 0; hop < 128 && v; ++hop) {
    auto* def = v.getDefiningOp();
    if (!def) return {};
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
      v = rb.getBox();
      continue;
    }
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(def)) {
      v = eb.getMemref();
      continue;
    }
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = cv.getValue();
      continue;
    }
    if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
      v = dg.getMemref();
      continue;
    }
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
      return decl;
    }
    return {};
  }
  return {};
}

std::optional<Rebind> matchRebindStore(fir::StoreOp store, const Candidate& cand) {
  auto memberSlot = mlir::dyn_cast_or_null<hlfir::DesignateOp>(store.getMemref().getDefiningOp());
  if (!memberSlot) return std::nullopt;
  auto memberOpt = memberSlot.getComponent();
  if (!memberOpt.has_value()) return std::nullopt;
  auto memberName = memberOpt->getValue();
  if (memberName.empty()) return std::nullopt;
  auto elemRef = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memberSlot.getMemref().getDefiningOp());
  if (!elemRef) return std::nullopt;
  auto srcDecl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(elemRef.getMemref().getDefiningOp());
  if (!srcDecl || srcDecl != cand.aosDecl) return std::nullopt;
  auto idxs = elemRef.getIndices();
  if (idxs.size() != 1) return std::nullopt;
  // ``recv_dp(icount)%p => ...`` rebinds index via an incremented counter, so
  // the element index is a forwarded store-then-load literal, not a bare
  // arith.constant -- resolve it through the counter's nearest store.
  auto outerC = constIntForwarded(idxs[0]);
  if (!outerC) return std::nullopt;
  Rebind r;
  r.outerIdx = *outerC;
  r.memberName = memberName.str();
  r.store = store;
  r.boxValue = store.getValue();
  r.targetDecl = traceTarget(store.getValue());
  if (!r.targetDecl) return std::nullopt;
  return r;
}

/// Set ``s.lbounds`` / ``s.lboundVals`` to the lower bounds the member takes over at a rebind: those of the box the
/// rebind stores.  Flang builds that box from the target: ``fir.embox %t(%shape_shift)`` for an explicit-shape target
/// or a bounds remap (``q(c)%x(5:, 7:) => t``), ``fir.rebox %t(%shift)`` for an assumed-shape target with lower bounds,
/// and a ``fir.rebox`` without a shift (every dim at 1) of ``hlfir.designate %t(lb:ub:1, ...)`` for ``=> t(:, :)``.
/// Returns false -- the candidate is then left as it is -- when the box is not provably all of ``targetDecl`` with
/// its extents: a part of the target (``=> t(2:5, :)``), a rank or extent remap, or a form this does not model.
bool setPointerLowerBounds(mlir::Value box, hlfir::DeclareOp targetDecl, InnerShape& s) {
  size_t const rank = s.shape.size();
  while (auto cv = box.getDefiningOp<fir::ConvertOp>()) box = cv.getValue();
  mlir::Value shape;
  mlir::Value source;
  if (auto eb = box.getDefiningOp<fir::EmboxOp>()) {
    if (eb.getSlice() || !keepsExtents(eb.getShape(), targetDecl, rank)) return false;
    shape = eb.getShape();
    source = eb.getMemref();
  } else if (auto rb = box.getDefiningOp<fir::ReboxOp>()) {
    if (rb.getSlice() || !keepsExtents(rb.getShape(), targetDecl, rank)) return false;
    shape = rb.getShape();
    source = rb.getBox();
  } else {
    return false;
  }
  while (true) {
    if (auto cv = source.getDefiningOp<fir::ConvertOp>())
      source = cv.getValue();
    else if (auto dg = source.getDefiningOp<hlfir::DesignateOp>(); dg && isWholeSection(dg))
      source = dg.getMemref();
    else
      break;
  }
  if (source.getDefiningOp<hlfir::DeclareOp>() != targetDecl) return false;

  s.lbounds.assign(rank, 1);
  s.lboundVals.assign(rank, mlir::Value{});
  if (isUnitShape(shape)) return true;
  auto origins = shapeOrigins(shape);
  if (origins.size() != rank) return false;
  for (size_t d = 0; d < rank; ++d) {
    if (auto c = constIntThroughConvert(origins[d])) {
      s.lbounds[d] = *c;
    } else {
      s.lbounds[d] = std::nullopt;
      s.lboundVals[d] = origins[d];
    }
  }
  return true;
}

/// True when two views give the member the same lower bounds: equal static bounds, or runtime ones that provably
/// hold the same value.
bool sameLowerBounds(const InnerShape& a, const InnerShape& b) {
  if (a.lbounds.size() != b.lbounds.size()) return false;
  for (size_t d = 0; d < a.lbounds.size(); ++d) {
    if (a.lbounds[d] || b.lbounds[d] ? a.lbounds[d] != b.lbounds[d] : !sameIndexValue(a.lboundVals[d], b.lboundVals[d]))
      return false;
  }
  return true;
}

/// True when a variable whose lower bounds ``shape`` gives (none / ``fir.shape``: every dim at 1) starts each dim
/// where the member does, so its element subscripts index the concat unchanged.
bool boundsMatchMember(mlir::Value shape, const InnerShape& s) {
  if (isUnitShape(shape)) return s.unitLowerBounds();
  auto origins = shapeOrigins(shape);
  if (origins.size() != s.lbounds.size()) return false;
  for (size_t d = 0; d < origins.size(); ++d) {
    if (s.lbounds[d] ? constIntThroughConvert(origins[d]) != s.lbounds[d]
                     : !sameIndexValue(origins[d], s.lboundVals[d]))
      return false;
  }
  return true;
}

// --------------------------------------------------------------------------
// Materialisation helpers.
// --------------------------------------------------------------------------

/// ``v`` as an ``index`` value, converted at the builder's insertion point when it is another integer type.
mlir::Value toIndex(mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) {
  return v.getType() == b.getIndexType() ? v : b.create<fir::ConvertOp>(loc, b.getIndexType(), v).getResult();
}

/// Inner dim ``d``'s lower bound as an index value at the builder's insertion point.
mlir::Value lowerBoundValue(mlir::OpBuilder& b, mlir::Location loc, const InnerShape& s, size_t d) {
  if (s.lbounds[d]) return b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(*s.lbounds[d])).getResult();
  assert(s.lboundVals[d] && "a lower bound is either static or an SSA value");
  return toIndex(b, loc, s.lboundVals[d]);
}

/// Create a top-level static-shape concat transient ``q_<member>`` at the
/// start of ``func``.  Returns the resulting ``hlfir.declare`` (whose
/// ``result(0)`` is the storage view used by all designates).
hlfir::DeclareOp createConcatStorage(mlir::OpBuilder& b, mlir::Location loc, mlir::func::FuncOp func, Candidate& cand,
                                     const MemberSpec& member, const InnerShape& innerShape,
                                     mlir::Operation* insertAfter, int64_t allocId) {
  mlir::OpBuilder::InsertionGuard const g(b);
  if (insertAfter)
    b.setInsertionPointAfter(insertAfter);
  else
    b.setInsertionPointToStart(&func.getBody().front());

  // An EMPTY AoS (``cand.N == 0``) allocates a size-1 PLACEHOLDER outer dim: a
  // zero-extent transient trips DaCe's memlet/shape validation, and every use
  // of this storage is in a dead 0-trip loop, so a 1-record placeholder that is
  // never touched is the safe registration.
  int64_t const allocN = cand.N > 0 ? cand.N : 1;
  llvm::SmallVector<int64_t, 4> typeDims;
  typeDims.push_back(allocN);
  for (auto d : innerShape.shape) typeDims.push_back(d);
  auto seqTy = fir::SequenceType::get(typeDims, innerShape.elemTy);

  // Per-dim extents as index Values.  Static dims synth an index
  // constant; dynamic dims re-use the captured SSA extent, or fall
  // back to ``fir.box_dims`` on the captured ``boxSource``.
  llvm::SmallVector<mlir::Value, 4> extents;
  auto outerC = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(allocN)).getResult();
  extents.push_back(outerC);
  auto idxTy = b.getIndexType();
  for (size_t d = 0; d < innerShape.shape.size(); ++d) {
    if (innerShape.shape[d] != fir::SequenceType::getUnknownExtent()) {
      auto cst = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(innerShape.shape[d]));
      extents.push_back(cst.getResult());
    } else if (innerShape.extentVals[d]) {
      mlir::Value ext = innerShape.extentVals[d];
      if (ext.getType() != idxTy) ext = b.create<fir::ConvertOp>(loc, idxTy, ext).getResult();
      extents.push_back(ext);
    } else if (innerShape.boxSource) {
      auto dimC = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(d)).getResult();
      auto bd = b.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, innerShape.boxSource, dimC);
      // bd.getResult(1) is the extent of dim ``d``.
      extents.push_back(bd.getResult(1));
    } else {
      return {};
    }
  }
  // Alloca shape: static when fully-static, dynamic-shape signature
  // otherwise.
  fir::AllocaOp alloca;
  if (innerShape.allStatic()) {
    alloca = b.create<fir::AllocaOp>(loc, seqTy);
  } else {
    alloca = b.create<fir::AllocaOp>(loc, seqTy, /*uniqName=*/llvm::StringRef{},
                                     /*bindcName=*/llvm::StringRef{},
                                     /*typeparams=*/mlir::ValueRange{},
                                     /*shape=*/mlir::ValueRange{extents});
  }
  mlir::Value shapeOper;
  if (innerShape.unitLowerBounds()) {
    shapeOper = b.create<fir::ShapeOp>(loc, extents).getResult();
  } else {
    // The inner dims keep the member's lower bounds, so its original
    // subscripts index the concat unchanged; the outer dim stays 1-based.
    llvm::SmallVector<mlir::Value, 10> pairs{b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(1)).getResult(),
                                             outerC};
    for (size_t d = 0; d < innerShape.shape.size(); ++d) {
      pairs.push_back(lowerBoundValue(b, loc, innerShape, d));
      pairs.push_back(extents[d + 1]);
    }
    auto shapeShiftTy = fir::ShapeShiftType::get(b.getContext(), extents.size());
    shapeOper = b.create<fir::ShapeShiftOp>(loc, shapeShiftTy, pairs).getResult();
  }
  auto uniqName = "_QXaos_lift_" + cand.aosDecl.getUniqName().str() + "_" + member.name + "_" + std::to_string(allocId);
  auto decl = b.create<hlfir::DeclareOp>(loc, alloca.getResult(), uniqName, shapeOper);
  return decl;
}

/// Emit a nested ``fir.do_loop`` tree that copies between two arrays of
/// shape ``innerShape``, with an additional fixed outer index for one
/// side.
///
/// ``directionCopyIn = true``  copies ``target(i...)`` into ``concat(c, i...)``
/// ``directionCopyIn = false`` copies ``concat(c, i...)`` into ``target(i...)``
///
/// The loops run 1 .. extent.  The concat starts each inner dim at the member's
/// lower bound, the target at the one its declare gives it; the two differ for a
/// bounds remap (``q(c)%x(5:, 7:) => t``), so each side's subscript is offset by
/// its own lower bound minus 1.  The concat is sized by the member's first
/// rebind; a dynamic extent not provably equal to the concat's is clamped to it,
/// so a target whose runtime extent differs is never copied past either end.
void emitCopyLoop(mlir::OpBuilder& b, mlir::Location loc, mlir::Operation* insertBefore, hlfir::DeclareOp concatDecl,
                  int64_t outerC, hlfir::DeclareOp targetDecl, const InnerShape& innerShape, bool directionCopyIn) {
  mlir::OpBuilder::InsertionGuard const g(b);
  b.setInsertionPoint(insertBefore);
  auto idxTy = b.getIndexType();
  auto c1 = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(1)).getResult();
  // Outer index as a literal: the concat's outer dim starts at 1.
  auto outerVal = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(outerC)).getResult();

  // Per-dim subscript offsets (lower bound - 1), null for a dim that starts at 1.
  size_t const rank = innerShape.shape.size();
  auto targetOrigins = shapeOrigins(targetDecl.getShape());
  auto concatExtents = shapeExtents(concatDecl.getShape());  // outer dim first
  assert((targetOrigins.empty() || targetOrigins.size() == rank) && concatExtents.size() == rank + 1);
  llvm::SmallVector<mlir::Value, 4> concatOffsets(rank);
  llvm::SmallVector<mlir::Value, 4> targetOffsets(rank);
  for (size_t d = 0; d < rank; ++d) {
    if (innerShape.lbounds[d] != 1)
      concatOffsets[d] = b.create<mlir::arith::SubIOp>(loc, lowerBoundValue(b, loc, innerShape, d), c1).getResult();
    if (!targetOrigins.empty() && constIntThroughConvert(targetOrigins[d]) != 1)
      targetOffsets[d] = b.create<mlir::arith::SubIOp>(loc, toIndex(b, loc, targetOrigins[d]), c1).getResult();
  }

  llvm::SmallVector<mlir::Value, 4> ivs;
  for (size_t d = 0; d < rank; ++d) {
    mlir::Value hi;
    if (innerShape.shape[d] != fir::SequenceType::getUnknownExtent()) {
      hi = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(innerShape.shape[d])).getResult();
    } else if (innerShape.extentVals[d]) {
      hi = toIndex(b, loc, innerShape.extentVals[d]);
    } else if (innerShape.boxSource) {
      auto dimC = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(d)).getResult();
      auto bd = b.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, innerShape.boxSource, dimC);
      hi = bd.getResult(1);
    } else {
      llvm_unreachable("an inner dim has a static extent, an SSA extent or a box to query");
    }
    if (!sameIndexValue(hi, concatExtents[d + 1]))
      hi = b.create<mlir::arith::MinSIOp>(loc, hi, concatExtents[d + 1]).getResult();
    auto loop = b.create<fir::DoLoopOp>(loc, c1, hi, c1,
                                        /*unordered=*/false,
                                        /*finalCountValue=*/false);
    ivs.push_back(loop.getInductionVar());
    b.setInsertionPointToStart(loop.getBody());
  }

  // Designates: full-rank element accesses.  The target uses its subscripts
  // only; the concat prepends the outer literal.
  auto subscript = [&](mlir::Value iv, mlir::Value offset) {
    return offset ? b.create<mlir::arith::AddIOp>(loc, iv, offset).getResult() : iv;
  };
  llvm::SmallVector<mlir::Value, 5> concatIdxs;
  llvm::SmallVector<mlir::Value, 5> targetIdxs;
  concatIdxs.push_back(outerVal);
  for (size_t d = 0; d < ivs.size(); ++d) {
    concatIdxs.push_back(subscript(ivs[d], concatOffsets[d]));
    targetIdxs.push_back(subscript(ivs[d], targetOffsets[d]));
  }

  auto elemRefTy = fir::ReferenceType::get(innerShape.elemTy);
  mlir::Value const srcRef = directionCopyIn ? targetDecl.getResult(0) : concatDecl.getResult(0);
  mlir::Value const dstRef = directionCopyIn ? concatDecl.getResult(0) : targetDecl.getResult(0);
  auto const& srcIdxs = directionCopyIn ? targetIdxs : concatIdxs;
  auto const& dstIdxs = directionCopyIn ? concatIdxs : targetIdxs;

  auto srcDg = b.create<hlfir::DesignateOp>(loc, elemRefTy, srcRef, mlir::ValueRange{srcIdxs});
  auto loaded = b.create<fir::LoadOp>(loc, srcDg.getResult());
  auto dstDg = b.create<hlfir::DesignateOp>(loc, elemRefTy, dstRef, mlir::ValueRange{dstIdxs});
  // ``hlfir.assign %loaded_scalar to %dst_element_ref`` -- a scalar
  // store into one element.  The bridge's ``buildAssignNode`` routes
  // this through the normal SDFG-emit path; a raw ``fir.store``
  // surfaces an access node whose data field resolves to the loop
  // iterator symbol instead of the destination array (a pre-existing
  // bridge gap), so the assign form is the right shape.
  b.create<hlfir::AssignOp>(loc, loaded.getResult(), dstDg.getResult(),
                            /*realloc=*/false,
                            /*keep_lhs_length_if_realloc=*/false,
                            /*temporary_lhs=*/false);
}

/// Rewrite one element designate ``q(idx)%m(i, j, ...)`` (reached through the box
/// load) to a direct scalar designate over ``concat(idx, i, j, ...)``.  Scalar
/// subscripts only -- safe to feed loop bounds / conditions / interstate edges.
void rewriteAccess(mlir::OpBuilder& b, hlfir::DesignateOp innerDg, mlir::Value outerIdxVal,
                   hlfir::DeclareOp concatDecl) {
  mlir::OpBuilder::InsertionGuard const g(b);
  b.setInsertionPoint(innerDg);
  auto idxTy = b.getIndexType();
  mlir::Value outerCast = outerIdxVal;
  if (outerCast.getType() != idxTy)
    outerCast = b.create<fir::ConvertOp>(innerDg.getLoc(), idxTy, outerCast).getResult();
  llvm::SmallVector<mlir::Value, 5> newIdxs;
  newIdxs.push_back(outerCast);
  for (auto idx : innerDg.getIndices()) newIdxs.push_back(idx);
  auto newDg = b.create<hlfir::DesignateOp>(innerDg.getLoc(), innerDg.getResult().getType(), concatDecl.getResult(0),
                                            mlir::ValueRange{newIdxs});
  innerDg.getResult().replaceAllUsesWith(newDg.getResult());
}

/// Build a box of the gather-buffer slice ``concat(idx, lb1:lb1+d1-1:1, ...)``
/// -- the idx-th rebind target's worth of the concat, rebased to the member's
/// lower bounds when they are not 1 -- typed to match the original
/// pointer box ``loadBoxTy`` so it can replace a ``fir.load`` of ``q(idx)%member``
/// directly.  Every whole-box use (element designate, ``fir.box_dims`` shape
/// query, ``fir.rebox``) then retargets to the concat with no per-use rewrite.
mlir::Value buildConcatSliceBox(mlir::OpBuilder& b, mlir::Location loc, hlfir::DeclareOp concatDecl,
                                mlir::Value outerIdx, mlir::Type loadBoxTy) {
  // The original access is a box load, so the type is a box; bail (no rewrite)
  // if not.  Reuse the concat declare's already-resolved per-dim extents: the
  // leading dim is the outer N, the rest the inner shape, with any dynamic inner
  // extent already materialised (as ``box_dims`` on the rebind target) at the
  // concat's creation site -- which dominates every read site we rewrite here.
  auto loadBox = mlir::dyn_cast<fir::BoxType>(loadBoxTy);
  if (!loadBox) return {};
  // The concat's per-dim extents, and its lower bounds when it keeps the targets' (``fir.shape_shift``).
  auto allExtents = shapeExtents(concatDecl.getShape());
  auto allOrigins = shapeOrigins(concatDecl.getShape());
  if (allExtents.size() < 2) return {};
  mlir::Value const outerCast = toIndex(b, loc, outerIdx);
  mlir::Value const one = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(1)).getResult();
  using Subscript = std::variant<mlir::Value, std::tuple<mlir::Value, mlir::Value, mlir::Value>>;
  llvm::SmallVector<Subscript, 5> subs;
  subs.push_back(outerCast);  // scalar outer index -> drops the N dim
  llvm::SmallVector<mlir::Value, 4> extents;
  llvm::SmallVector<mlir::Value, 4> lowerBounds;
  for (unsigned d = 1; d < allExtents.size(); ++d) {  // skip the leading outer-N dim
    mlir::Value const ext = toIndex(b, loc, allExtents[d]);
    extents.push_back(ext);
    if (allOrigins.empty()) {
      subs.push_back(std::make_tuple(one, ext, one));  // full inner range 1:ext:1
    } else {
      // full inner range lb:lb+ext-1:1
      mlir::Value const lb = toIndex(b, loc, allOrigins[d]);
      auto last = b.create<mlir::arith::SubIOp>(loc, ext, one).getResult();
      subs.push_back(std::make_tuple(lb, b.create<mlir::arith::AddIOp>(loc, lb, last).getResult(), one));
      lowerBounds.push_back(lb);
    }
  }
  mlir::Value const shape = b.create<fir::ShapeOp>(loc, extents).getResult();
  mlir::Type innerEle = loadBox.getEleTy();
  if (auto pt = mlir::dyn_cast<fir::PointerType>(innerEle))
    innerEle = pt.getEleTy();
  else if (auto ht = mlir::dyn_cast<fir::HeapType>(innerEle))
    innerEle = ht.getEleTy();
  auto sliceBoxTy = fir::BoxType::get(innerEle);
  mlir::Value sliceBox = b.create<hlfir::DesignateOp>(loc, sliceBoxTy, concatDecl.getResult(0), llvm::StringRef{},
                                                      mlir::Value{}, subs, mlir::ValueRange{}, std::optional<bool>{},
                                                      shape, mlir::ValueRange{}, fir::FortranVariableFlagsAttr{})
                             .getResult();
  if (!lowerBounds.empty()) {
    // A section starts at 1, the member keeps its own lower bounds (what ``LBOUND`` and box-relative subscripts of
    // the replaced pointer box see).
    auto shiftTy = fir::ShiftType::get(b.getContext(), lowerBounds.size());
    mlir::Value const shift = b.create<fir::ShiftOp>(loc, shiftTy, lowerBounds).getResult();
    sliceBox = b.create<fir::ReboxOp>(loc, sliceBoxTy, sliceBox, shift, /*slice=*/mlir::Value{}).getResult();
  }
  if (sliceBox.getType() != loadBoxTy) sliceBox = b.create<fir::ConvertOp>(loc, loadBoxTy, sliceBox).getResult();
  return sliceBox;
}

/// True when element subscripts on ``rb``'s result index the concat's inner dims unchanged: the rebox takes no section,
/// does not reshape, and starts every dim where the concat does (a rebox without a shift starts them at 1).
bool reboxKeepsConcatIndexing(fir::ReboxOp rb, hlfir::DeclareOp concatDecl) {
  mlir::Value const shape = rb.getShape();
  if (rb.getSlice() || (shape && !shape.getDefiningOp<fir::ShiftOp>())) return false;
  auto isOne = [](mlir::Value v) { return constIntThroughConvert(v) == 1; };
  auto concatOrigins = shapeOrigins(concatDecl.getShape());  // outer dim first; empty when every dim starts at 1
  auto reboxOrigins = shapeOrigins(shape);
  if (concatOrigins.empty()) return llvm::all_of(reboxOrigins, isOne);
  if (reboxOrigins.empty()) return llvm::all_of(llvm::drop_begin(concatOrigins), isOne);
  if (reboxOrigins.size() + 1 != concatOrigins.size()) return false;
  for (size_t d = 0; d < reboxOrigins.size(); ++d)
    if (!sameIndexValue(reboxOrigins[d], concatOrigins[d + 1])) return false;
  return true;
}

/// Replace every box load of ``q(idx)%member`` with a slice box of the gather
/// buffer (``concat(idx, ...)``).  Element designates, shape queries and reboxes
/// over that box then all read the concat; the now-dead load / member-slot /
/// element-designate chain is swept by ``eraseDeadAosChain``.  Reads reach the
/// slot directly or behind a re-``hlfir.declare`` / ``fir.rebox`` / ``fir.embox`` /
/// ``fir.convert`` alias the fresh flang IR interposes -- follow those forward.
unsigned rewriteAllAccesses(mlir::OpBuilder& b, Candidate& cand,
                            const llvm::DenseMap<llvm::StringRef, hlfir::DeclareOp>& concatByMember) {
  unsigned rewritten = 0;
  llvm::SmallVector<hlfir::DesignateOp, 8> elemDesignates;
  llvm::SmallVector<mlir::Value, 4> aliasRoots{cand.aosDecl.getResult(0), cand.aosDecl.getResult(1)};
  llvm::SmallPtrSet<mlir::Operation*, 8> seenAlias;
  for (size_t ai = 0; ai < aliasRoots.size(); ++ai) {
    mlir::Value const root = aliasRoots[ai];
    if (!root) continue;
    for (auto* u : root.getUsers()) {
      if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u)) {
        if (dg.getIndices().size() == 1) elemDesignates.push_back(dg);
      } else if (auto rd = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
        if (seenAlias.insert(rd).second) {
          aliasRoots.push_back(rd.getResult(0));
          aliasRoots.push_back(rd.getResult(1));
        }
      } else if (auto rb = mlir::dyn_cast<fir::ReboxOp>(u)) {
        if (seenAlias.insert(rb).second) aliasRoots.push_back(rb.getResult());
      } else if (auto eb = mlir::dyn_cast<fir::EmboxOp>(u)) {
        if (seenAlias.insert(eb).second) aliasRoots.push_back(eb.getResult());
      } else if (auto cv = mlir::dyn_cast<fir::ConvertOp>(u)) {
        if (seenAlias.insert(cv).second) aliasRoots.push_back(cv.getResult());
      }
    }
  }
  for (auto elemDg : elemDesignates) {
    mlir::Value const outerIdx = elemDg.getIndices()[0];
    llvm::SmallVector<hlfir::DesignateOp, 4> memberDgs;
    for (auto* u : elemDg.getResult().getUsers())
      if (auto md = mlir::dyn_cast<hlfir::DesignateOp>(u)) memberDgs.push_back(md);
    for (auto memberDg : memberDgs) {
      auto memberOpt = memberDg.getComponent();
      if (!memberOpt.has_value()) continue;
      auto memberName = memberOpt->getValue();
      auto it = concatByMember.find(memberName);
      if (it == concatByMember.end()) continue;
      llvm::SmallVector<fir::LoadOp, 4> loads;
      for (auto* u : memberDg.getResult().getUsers())
        if (auto ld = mlir::dyn_cast<fir::LoadOp>(u)) loads.push_back(ld);
      for (auto load : loads) {
        // Step 1: rewrite the element designates (scalar) reached through the box
        // load -- directly or behind a ``fir.box_addr`` / ``fir.rebox`` /
        // ``fir.convert`` reinterpret -- to direct designates over the concat.
        // Scalar subscripts keep loop bounds and conditions (interstate edges)
        // range-free, so element-only kernels (Graupel ``q``) need no view.  Only
        // reinterprets that keep the concat's indexing are looked through; the
        // others stay whole-box uses of the slice box built in step 2.
        hlfir::DeclareOp concatDecl = it->second;
        bool const unitConcat = isUnitShape(concatDecl.getShape());
        llvm::SmallVector<hlfir::DesignateOp, 4> innerDgs;
        llvm::SmallVector<mlir::Operation*, 8> peelOps;
        llvm::SmallVector<mlir::Value, 4> boxVals{load.getResult()};
        llvm::SmallPtrSet<mlir::Operation*, 8> seenBox;
        for (size_t bi = 0; bi < boxVals.size(); ++bi)
          for (auto* u : boxVals[bi].getUsers()) {
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u)) {
              innerDgs.push_back(dg);
            } else if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(u); ba && unitConcat) {
              if (seenBox.insert(ba).second) {
                boxVals.push_back(ba.getResult());
                peelOps.push_back(ba);
              }
            } else if (auto rb = mlir::dyn_cast<fir::ReboxOp>(u); rb && reboxKeepsConcatIndexing(rb, concatDecl)) {
              if (seenBox.insert(rb).second) {
                boxVals.push_back(rb.getResult());
                peelOps.push_back(rb);
              }
            } else if (auto cv = mlir::dyn_cast<fir::ConvertOp>(u)) {
              if (seenBox.insert(cv).second) {
                boxVals.push_back(cv.getResult());
                peelOps.push_back(cv);
              }
            }
          }
        for (auto innerDg : innerDgs) {
          rewriteAccess(b, innerDg, outerIdx, it->second);
          innerDg.erase();
          rewritten++;
        }
        // Erase the now-dead reinterpret chain (innermost first) so the load's
        // surviving users are only genuine whole-box uses.
        for (int pi = static_cast<int>(peelOps.size()) - 1; pi >= 0; --pi)
          if (peelOps[pi]->use_empty()) peelOps[pi]->erase();
        // Step 2: any surviving whole-box use -- a ``fir.box_dims`` shape query or
        // a whole-array rebox/pass -- cannot be an element designate, so re-point
        // it at a concat slice box.  Element-only loads are use-empty here, so no
        // range view is created for them (keeps Graupel ``q`` conditions scalar).
        if (!load.getResult().use_empty()) {
          b.setInsertionPointAfter(load);
          mlir::Value const sliceBox =
              buildConcatSliceBox(b, load.getLoc(), it->second, outerIdx, load.getResult().getType());
          if (sliceBox) {
            load.getResult().replaceAllUsesWith(sliceBox);
            rewritten++;
          }
        }
      }
    }
  }
  return rewritten;
}

/// After the gather replaces every read with a concat slice box and the copy-in
/// erases the rebind stores, the AoS pointer array's designate / load / box
/// chain is dead.  Sweep the use-less VALUE-producing ops in the forest rooted at
/// the AoS declare so the lowered SDFG never mints the flattened
/// ``<aos>_<member>`` symbol those dead designates would surface.  Side-effecting
/// stores (no results, so trivially ``use_empty``) are skipped -- the copy-in
/// already erased the rebinds, and sweeping a store would double-free it.
void eraseDeadAosChain(Candidate& cand) {
  llvm::SmallPtrSet<mlir::Operation*, 32> seen;
  llvm::SmallVector<mlir::Operation*, 32> forest{cand.aosDecl.getOperation()};
  seen.insert(cand.aosDecl.getOperation());
  llvm::SmallVector<mlir::Value, 8> roots{cand.aosDecl.getResult(0), cand.aosDecl.getResult(1)};
  for (size_t i = 0; i < roots.size(); ++i) {
    if (!roots[i]) continue;
    for (auto* u : roots[i].getUsers())
      if (seen.insert(u).second) {
        forest.push_back(u);
        for (auto res : u->getResults()) roots.push_back(res);
      }
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto*& op : forest)
      if (op && op->getNumResults() > 0 && op->use_empty()) {
        op->erase();
        op = nullptr;
        changed = true;
      }
  }
}

// --------------------------------------------------------------------------
// Direct accesses to a rebind target while its alias lives in the concat.
// --------------------------------------------------------------------------

/// The accesses to one rebind's target between the rebind and the end of the rebind's block, the window in which
/// the concat holds the live values.
struct TargetWindow {
  /// Element designates ``target(i, j)``, redirected to ``concat(c, i, j)``.
  llvm::SmallVector<hlfir::DesignateOp, 8> elements;
  /// Any other access; the concat and the target are synchronised around it.
  llvm::SmallVector<mlir::Operation*, 4> others;
  /// An access that neither redirect nor synchronisation can serve.
  mlir::Operation* unsupported = nullptr;
};

/// True when a designate selects one element by scalar subscripts alone: no section, component or substring.
bool isElementDesignate(hlfir::DesignateOp dg) {
  if (dg.getComponent().has_value() || !dg.getTypeparams().empty()) return false;
  for (bool const triplet : dg.getIsTriplet())
    if (triplet) return false;
  return mlir::isa<fir::ReferenceType>(dg.getResult().getType());
}

/// Collect the accesses to ``r.targetDecl`` that follow the rebind store in its block, looking through the
/// re-declares, reboxes and converts that alias it (an inlined callee's dummy argument).  ``member`` is the rebind's
/// view of the member, whose lower bounds the concat keeps.  Call BEFORE the store is erased.
TargetWindow collectTargetWindow(const Rebind& r, const InnerShape& member) {
  TargetWindow w;
  mlir::Block* block = r.store->getBlock();
  auto inWindow = [&](mlir::Operation* user) {
    auto* anc = block->findAncestorOpInBlock(*user);
    return anc && r.store->isBeforeInBlock(anc);
  };
  // Per alias of the target: whether it keeps the target's element layout (no section or reshape on the way; a
  // re-declare is taken to keep it, an inlined dummy's own shape is not compared), and whether its lower bounds are
  // the member's.  Each declare, rebox and embox sets the bounds afresh: a declare's first result from its shape, its
  // second, the raw base, none at all.  An element designate indexes the concat unchanged only when both hold.
  struct View {
    bool keepsLayout;
    bool memberBounds;
  };
  llvm::SmallVector<mlir::Value, 8> roots;
  llvm::DenseMap<mlir::Value, View> views;
  llvm::SmallPtrSet<mlir::Operation*, 16> seen;
  auto addRoot = [&](mlir::Value v, View view) {
    roots.push_back(v);
    views[v] = view;
  };
  auto addDeclare = [&](hlfir::DeclareOp decl, bool keepsLayout) {
    addRoot(decl.getResult(0), {keepsLayout, boundsMatchMember(decl.getShape(), member)});
    addRoot(decl.getResult(1), {keepsLayout, isUnitShape(decl.getShape()) && member.unitLowerBounds()});
  };
  auto reshapes = [](mlir::Value shape) { return shape && !shape.getDefiningOp<fir::ShiftOp>(); };
  addDeclare(r.targetDecl, true);
  for (size_t i = 0; i < roots.size(); ++i) {
    mlir::Value const root = roots[i];
    View const view = views[root];
    for (auto* u : root.getUsers()) {
      if (auto rd = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
        if (seen.insert(rd).second) addDeclare(rd, view.keepsLayout);
      } else if (auto rb = mlir::dyn_cast<fir::ReboxOp>(u)) {
        if (seen.insert(rb).second)
          addRoot(rb.getResult(), {view.keepsLayout && !rb.getSlice() && !reshapes(rb.getShape()),
                                   boundsMatchMember(rb.getShape(), member)});
      } else if (auto eb = mlir::dyn_cast<fir::EmboxOp>(u)) {
        if (seen.insert(eb).second)
          addRoot(eb.getResult(), {view.keepsLayout && !eb.getSlice(), boundsMatchMember(eb.getShape(), member)});
      } else if (auto cv = mlir::dyn_cast<fir::ConvertOp>(u)) {
        if (seen.insert(cv).second) addRoot(cv.getResult(), view);
      } else if (!inWindow(u) || mlir::isa<fir::BoxDimsOp>(u)) {
        continue;  // outside the window, or a shape query: it never touches the data
      } else if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u)) {
        if (dg.getMemref() != root) continue;  // a subscript operand, not the designated array
        if (isElementDesignate(dg) && view.keepsLayout && view.memberBounds)
          w.elements.push_back(dg);
        else if (!w.unsupported)
          w.unsupported = dg;  // a section aliases the target past this op; other subscripts miss the concat
      } else if (seen.insert(u).second) {
        w.others.push_back(u);
      }
    }
  }
  return w;
}

/// Redirect and synchronise the window's accesses to the concat.  Call after the copy-in is emitted and before the
/// final copy-out is.
void applyTargetWindow(mlir::OpBuilder& b, const Rebind& r, const TargetWindow& w, hlfir::DeclareOp concatDecl,
                       const InnerShape& shape) {
  for (auto dg : w.elements) {
    mlir::OpBuilder::InsertionGuard const g(b);
    b.setInsertionPoint(dg);
    auto outer = b.create<mlir::arith::ConstantOp>(dg.getLoc(), b.getIndexAttr(r.outerIdx)).getResult();
    rewriteAccess(b, dg, outer, concatDecl);
    dg.erase();
  }
  for (auto* op : w.others) {
    // The op reads and may write the target itself: bring the target up to date first, the concat after.
    emitCopyLoop(b, op->getLoc(), op, concatDecl, r.outerIdx, r.targetDecl, shape, /*directionCopyIn=*/false);
    if (auto* next = op->getNextNode())
      emitCopyLoop(b, op->getLoc(), next, concatDecl, r.outerIdx, r.targetDecl, shape, /*directionCopyIn=*/true);
  }
}

// --------------------------------------------------------------------------
// Pass.
// --------------------------------------------------------------------------

struct LiftAosPointerRecordsPass
    : public mlir::PassWrapper<LiftAosPointerRecordsPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftAosPointerRecordsPass)

  llvm::StringRef getArgument() const final { return "hlfir-lift-aos-pointer-records"; }
  llvm::StringRef getDescription() const final {
    return "Lift AoS-of-records-with-pointer-only members to flat per-"
           "member concatenation transients with copy-in / copy-out.";
  }

  void runOnOperation() override {
    auto module = getOperation();
    for (auto func : llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
      if (!processFunction(func)) return signalPassFailure();
    }
  }

  static bool processFunction(mlir::func::FuncOp func) {
    llvm::SmallVector<Candidate, 2> cands;
    func.walk([&](hlfir::DeclareOp d) {
      if (auto c = matchCandidate(d)) cands.push_back(*c);
    });
    int allocId = 0;
    for (auto& c : cands)
      if (!processCandidate(func, c, allocId)) return false;
    return true;
  }

  static bool processCandidate(mlir::func::FuncOp func, Candidate& cand, int& allocId) {
    llvm::SmallVector<Rebind, 8> rebinds;
    func.walk([&](fir::StoreOp store) {
      if (auto r = matchRebindStore(store, cand)) rebinds.push_back(*r);
    });
    if (rebinds.empty()) {
      // A matched AoS-of-pointer-records with NO rebinds is DEAD exchange
      // scaffolding: ICON ``recv_sp`` when ``nfields_sp`` folds to 0 (no
      // single-precision fields this config) is allocated but never rebound.
      // After ``hlfir-inline-all`` folds the exchange worker into the caller,
      // its ``recv_sp(k)%p`` reads survive as 0-trip-loop bodies (dead, never
      // executed) -- so the AoS declare is NOT ``use_empty`` and cannot simply
      // be erased.  Those dead accesses still emit ``<name>_p`` SDFG access
      // nodes, so the flat transient MUST be registered here -- otherwise
      // ``hlfir-flatten-structs`` mints a broken ``recv_sp_p`` companion for the
      // unsupported AoS-of-pointer shape and ``prune_unused_arrays`` KeyErrors
      // on the dangling access.  With no rebind target to derive the
      // assumed-shape inner extents from, size each member concat to a size-1
      // placeholder from the member box TYPE (``innerShapeFromMemberBox``); the
      // accesses are all dead, so the placeholder is never touched and
      // ``prune_unused_arrays`` then drops the whole concat as unused.
      mlir::OpBuilder b(func.getContext());
      llvm::DenseMap<llvm::StringRef, hlfir::DeclareOp> concatByMember;
      mlir::Operation* insertAfter = cand.aosDecl.getOperation();
      for (auto& spec : cand.members) {
        auto s = innerShapeFromMemberBox(spec.boxTy);
        if (!s) break;
        auto declare = createConcatStorage(b, func.getLoc(), func, cand, spec, *s, insertAfter, allocId++);
        if (!declare) break;
        concatByMember[llvm::StringRef(spec.name)] = declare;
        insertAfter = declare.getOperation();
      }
      if (concatByMember.size() == cand.members.size()) rewriteAllAccesses(b, cand, concatByMember);
      eraseDeadAosChain(cand);
      return true;
    }

    // DYNAMIC-extent AoS (matchCandidate couldn't fold the outer extent): size
    // the concat by the max 1-based rebind index -- the rebinds are the static
    // ``recv1_sp .. recvM_sp`` slots and the exchange loop index is bounded by
    // ``nfields_sp <= M``, so a static ``(M, inner)`` concat covers every access.
    if (cand.dynamicExtent) {
      int64_t maxIdx = 0;
      for (auto& r : rebinds) maxIdx = std::max<int64_t>(maxIdx, r.outerIdx);
      cand.N = maxIdx;
    }

    // Stamp the recognition attribute on the function unconditionally
    // -- this surfaces the matched candidates to debug tools and the
    // test suite regardless of whether materialisation runs (the
    // materialiser bails when shapes can't fold to static integers).
    {
      auto* ctx = func.getContext();
      std::string const key = ("hlfir.aos_ptr_records." + cand.aosDecl.getUniqName().str());
      llvm::SmallVector<mlir::Attribute, 8> entries;
      for (auto& r : rebinds) {
        llvm::SmallVector<mlir::NamedAttribute, 3> kv;
        kv.push_back(
            {mlir::StringAttr::get(ctx, "outer"), mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), r.outerIdx)});
        kv.push_back({mlir::StringAttr::get(ctx, "member"), mlir::StringAttr::get(ctx, r.memberName)});
        kv.push_back(
            {mlir::StringAttr::get(ctx, "target"), mlir::StringAttr::get(ctx, r.targetDecl.getUniqName().str())});
        entries.push_back(mlir::DictionaryAttr::get(ctx, kv));
      }
      func->setAttr(key, mlir::ArrayAttr::get(ctx, entries));
    }

    // Gather path: materialise a flat ``(N, inner...)`` concat transient and copy
    // each rebind target in / out, rewriting accesses to index it.  Required when a
    // read index is a runtime value (no single target to alias) -- e.g. a
    // runtime-indexed select over the rebound pointers.
    // Each rebind's view of the member: its target's extents and the lower
    // bounds of the box it stores.  The copy loops of a rebind use its own view;
    // the concat takes the member's FIRST view, so the rebinds of one member
    // must agree on it.  Lower bounds come from the source (a target's
    // declaration or a remap) and must provably agree; a runtime extent read
    // from a descriptor cannot be compared here, so only static extents are
    // checked and the runtime ones are taken to agree (the copy loops clamp to
    // the concat's).
    llvm::SmallVector<InnerShape, 8> rebindShapes;
    llvm::DenseMap<llvm::StringRef, InnerShape> shapeByMember;
    for (auto& r : rebinds) {
      auto s = innerShapeFromTargetDecl(r.targetDecl);
      if (!s || !setPointerLowerBounds(r.boxValue, r.targetDecl, *s)) return true;
      auto [recorded, first] = shapeByMember.try_emplace(llvm::StringRef(r.memberName), *s);
      if (!first && (s->shape != recorded->second.shape || !sameLowerBounds(*s, recorded->second))) return true;
      rebindShapes.push_back(std::move(*s));
    }

    mlir::OpBuilder b(func.getContext());
    // Find the latest declare that contributes a shape extent we depend
    // on, so the concat declare is inserted in a position that sees
    // every required SSA value.  At minimum the AoS declare itself
    // must be visible (so its uniq_name is in scope for our derived
    // name).  Targets dominated by their hlfir.declare ops feed the
    // concat's per-dim extents.  A value from an enclosing block is visible
    // already.  The concat must precede every rebind: a value defined in a
    // nested block, or after a rebind (a remap's runtime lower bound computed
    // at a later rebind of another member), leaves the candidate as it is.
    mlir::DominanceInfo const dom(func);
    mlir::Block* aosBlock = cand.aosDecl->getBlock();
    mlir::Operation* insertAfter = cand.aosDecl.getOperation();
    bool hoistable = true;
    auto pushDep = [&](mlir::Value v) {
      auto* def = v ? v.getDefiningOp() : nullptr;
      if (!def) return;
      if (def->getBlock() == aosBlock) {
        if (insertAfter->isBeforeInBlock(def)) insertAfter = def;
      } else if (!dom.properlyDominates(def, cand.aosDecl)) {
        hoistable = false;
      }
    };
    for (auto& kv : shapeByMember) {
      for (auto v : kv.second.extentVals) pushDep(v);
      for (auto v : kv.second.lboundVals) pushDep(v);
      pushDep(kv.second.boxSource);
    }
    // Per-rebind: also account for the target declare itself, which is
    // the SSA producer of any box-source the copy loops will read.
    for (auto& r : rebinds) {
      pushDep(r.targetDecl.getResult(0));
      pushDep(r.targetDecl.getResult(1));
    }
    for (auto& r : rebinds) hoistable = hoistable && dom.properlyDominates(insertAfter, r.store);
    if (!hoistable) return true;
    llvm::DenseMap<llvm::StringRef, hlfir::DeclareOp> concatByMember;
    for (auto& spec : cand.members) {
      auto memberKey = llvm::StringRef(spec.name);
      auto sIt = shapeByMember.find(memberKey);
      if (sIt == shapeByMember.end()) continue;  // member never rebound
      auto declare = createConcatStorage(b, func.getLoc(), func, cand, spec, sIt->second, insertAfter, allocId++);
      if (!declare) return true;
      concatByMember[memberKey] = declare;
      insertAfter = declare.getOperation();
    }

    // Copy-in: at each rebind store's location, emit the loop nest; capture the
    // rebind's enclosing BLOCK (for the matching copy-out) BEFORE erasing the
    // store so the pointer slot is no longer written.
    llvm::SmallVector<mlir::Block*, 4> rebindBlock(rebinds.size(), nullptr);
    llvm::SmallVector<TargetWindow, 4> windows(rebinds.size());
    for (size_t i = 0; i < rebinds.size(); ++i) {
      auto& r = rebinds[i];
      auto it = concatByMember.find(r.memberName);
      if (it == concatByMember.end()) continue;
      auto const& shape = rebindShapes[i];
      rebindBlock[i] = r.store->getBlock();
      windows[i] = collectTargetWindow(r, shape);
      if (windows[i].unsupported) {
        windows[i].unsupported->emitError(
            "hlfir-lift-aos-pointer-records: cannot redirect this access to the pointer target ``" +
            r.targetDecl.getUniqName().str() + "`` while ``" + cand.aosDecl.getUniqName().str() + "(" +
            std::to_string(r.outerIdx) + ")%" + r.memberName +
            "`` points at it: it takes a section of the target, or its subscripts are relative to lower bounds other "
            "than the member's");
        return false;
      }
      emitCopyLoop(b, r.store.getLoc(), r.store, it->second, r.outerIdx, r.targetDecl, shape, /*directionCopyIn=*/true);
      r.store.erase();
    }

    // Rewrite access chains.
    rewriteAllAccesses(b, cand, concatByMember);

    // Between a rebind and its copy-out the concat holds the live values, so route the target's own accesses there.
    for (size_t i = 0; i < rebinds.size(); ++i) {
      if (!rebindBlock[i]) continue;
      auto& r = rebinds[i];
      applyTargetWindow(b, r, windows[i], concatByMember.find(r.memberName)->second, rebindShapes[i]);
    }

    // Copy-out: mirror copy-in, placed before the TERMINATOR of the rebind's own
    // block.  For a func-scope AoS that block is the entry block (terminator =
    // ``func.return`` -- identical to the previous placement).  For an AoS local
    // to a procedure INLINED into a nested region (ICON mult_mixprec's
    // ``recv_dp`` inside ``IF(my_process_is_mpi_parallel)``) it is the nested
    // region's end -- where the rebind targets (also nested) still dominate,
    // unlike a blanket ``func.return`` copy-out (which violates dominance).
    for (size_t i = 0; i < rebinds.size(); ++i) {
      auto& r = rebinds[i];
      if (!rebindBlock[i]) continue;
      auto it = concatByMember.find(r.memberName);
      if (it == concatByMember.end()) continue;
      mlir::Operation* term = rebindBlock[i]->getTerminator();
      emitCopyLoop(b, term->getLoc(), term, it->second, r.outerIdx, r.targetDecl, rebindShapes[i],
                   /*directionCopyIn=*/false);
    }
    eraseDeadAosChain(cand);
    return true;
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLiftAosPointerRecordsPass() { return std::make_unique<LiftAosPointerRecordsPass>(); }

}  // namespace hlfir_bridge
