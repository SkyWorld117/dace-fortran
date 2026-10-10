// ============================================================================
// RecordAccess.cpp  --  see RecordAccess.h.
// ============================================================================

#include "passes/RecordAccess.h"

#include "bridge/trace_utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace hlfir_bridge {

namespace {

/// The array type under references, boxes, heap and pointer wrappers; null for a scalar.
fir::SequenceType arrayType(mlir::Type t) {
  for (int i = 0; i < 8 && t; ++i) {
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(t)) return seq;
    if (auto ref = mlir::dyn_cast<fir::ReferenceType>(t))
      t = ref.getEleTy();
    else if (auto box = mlir::dyn_cast<fir::BoxType>(t))
      t = box.getEleTy();
    else if (auto heap = mlir::dyn_cast<fir::HeapType>(t))
      t = heap.getEleTy();
    else if (auto ptr = mlir::dyn_cast<fir::PointerType>(t))
      t = ptr.getEleTy();
    else
      return {};
  }
  return {};
}

unsigned arrayRank(mlir::Value v) {
  fir::SequenceType const seq = arrayType(v.getType());
  return seq ? seq.getDimension() : 0;
}

/// The shape operand of a declaration or box, null for anything else.
mlir::Value shapeOperand(mlir::Value v) {
  if (auto decl = v.getDefiningOp<hlfir::DeclareOp>()) return decl.getShape();
  if (auto box = v.getDefiningOp<fir::EmboxOp>()) return box.getShape();
  if (auto rebox = v.getDefiningOp<fir::ReboxOp>()) return rebox.getShape();
  return {};
}

/// The lower bounds in which subscripts of ``v`` are given, one per dim; a null value stands for 1.  Those of a
/// declaration or box come from its shape or shift (1 without one: an assumed-shape dummy, a rebox without a shift),
/// a conversion keeps its operand's, and a section starts at 1.
llvm::SmallVector<mlir::Value, 4> lowerBounds(mlir::Value v) {
  llvm::SmallVector<mlir::Value, 4> lbs(arrayRank(v));
  if (auto conv = v.getDefiningOp<fir::ConvertOp>()) return lowerBounds(conv.getValue());
  ShapeOperandInfo const si = classifyShapeOperand(shapeOperand(v));
  for (size_t k = 0; k < lbs.size() && k < si.lbs.size(); ++k) lbs[k] = si.lbs[k];
  return lbs;
}

/// The extents a declaration gives explicitly (fir.shape / fir.shape_shift), empty when it takes those of what it
/// declares (no shape, or only a shift).
std::vector<mlir::Value> explicitExtents(hlfir::DeclareOp decl) {
  return classifyShapeOperand(decl.getShape()).extents;
}

/// The extents of an array value where they are known: from its declaration's or box's shape, else from its type
/// (null for a dynamic extent).
llvm::SmallVector<mlir::Value, 4> knownExtents(mlir::Value v) {
  llvm::SmallVector<mlir::Value, 4> exts(arrayRank(v));
  if (auto conv = v.getDefiningOp<fir::ConvertOp>()) return knownExtents(conv.getValue());
  ShapeOperandInfo const si = classifyShapeOperand(shapeOperand(v));
  for (size_t k = 0; k < exts.size() && k < si.extents.size(); ++k) exts[k] = si.extents[k];
  return exts;
}

std::optional<int64_t> constantOr1(mlir::Value v) { return v ? traceConstInt(v) : std::optional<int64_t>(1); }

bool sameValue(mlir::Value a, mlir::Value b) {
  if (a == b) return true;
  auto ca = constantOr1(a);
  auto cb = constantOr1(b);
  return ca && cb && *ca == *cb;
}

/// True when a declaration with explicit extents over ``storage`` addresses it element by element in the same
/// positions: every dim but the last has the same extent (column-major order).  An assumed-shape declaration takes the
/// shape of what it declares.
bool sameLayout(hlfir::DeclareOp decl, mlir::Value storage) {
  std::vector<mlir::Value> const own = explicitExtents(decl);
  if (own.size() < 2) return true;
  llvm::SmallVector<mlir::Value, 4> const theirs = knownExtents(storage);
  for (size_t k = 0; k + 1 < own.size(); ++k)
    if (k >= theirs.size() || !theirs[k] || !sameValue(own[k], theirs[k])) return false;
  return true;
}

mlir::Value toIndex(mlir::OpBuilder& b, mlir::Location loc, mlir::Value v) {
  if (!v) return b.create<mlir::arith::ConstantIndexOp>(loc, 1).getResult();
  return v.getType().isIndex() ? v : b.create<fir::ConvertOp>(loc, b.getIndexType(), v).getResult();
}

/// ``u - from + to``, or ``u`` when the bounds agree.
mlir::Value rebase(mlir::OpBuilder& b, mlir::Location loc, mlir::Value u, mlir::Value from, mlir::Value to) {
  if (sameValue(from, to)) return u;
  auto shifted = b.create<mlir::arith::SubIOp>(loc, u, toIndex(b, loc, from));
  return b.create<mlir::arith::AddIOp>(loc, shifted, toIndex(b, loc, to)).getResult();
}

}  // namespace

mlir::Value viewedArray(mlir::Operation* op) {
  if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(op)) return decl.getMemref();
  if (auto box = mlir::dyn_cast<fir::EmboxOp>(op)) return box.getSlice() ? mlir::Value{} : box.getMemref();
  if (auto rebox = mlir::dyn_cast<fir::ReboxOp>(op)) return rebox.getSlice() ? mlir::Value{} : rebox.getBox();
  if (auto conv = mlir::dyn_cast<fir::ConvertOp>(op)) return conv.getValue();
  if (auto part = mlir::dyn_cast<hlfir::DesignateOp>(op)) {
    if (part.getComponentAttr() || part.getSubstring().size() != 0 || part.getComplexPart()) return {};
    if (llvm::none_of(part.getIsTriplet(), [](bool t) { return t; })) return {};  // an element
    return part.getMemref();
  }
  return {};
}

namespace {

/// The views from ``view`` to the first value ``isRoot`` accepts, access first.
std::optional<llvm::SmallVector<mlir::Operation*, 4>> traceViews(mlir::Value view,
                                                                 llvm::function_ref<bool(mlir::Value)> isRoot) {
  llvm::SmallVector<mlir::Operation*, 4> hops;
  for (int i = 0; i < 64 && view; ++i) {
    if (isRoot(view)) return hops;
    mlir::Operation* def = view.getDefiningOp();
    if (!def) return std::nullopt;
    mlir::Value const next = viewedArray(def);
    if (!next) return std::nullopt;
    if (!mlir::isa<hlfir::DesignateOp>(def)) {
      // A declaration or box over the same elements: same rank, and for an explicit shape the same layout.
      // Sequence association (another rank or shape over the same storage) is not modelled.
      if (arrayRank(view) != arrayRank(next)) return std::nullopt;
      if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(def); decl && !sameLayout(decl, next)) return std::nullopt;
    }
    hops.push_back(def);
    view = next;
  }
  return std::nullopt;
}

/// Compose ``subscripts`` of ``view`` through ``hops`` into the index space of the root, whose lower bounds are
/// ``rootLbs`` (a null value stands for 1).  The lower bounds of the views in between cancel out; only the first
/// view's and the root's matter.
llvm::SmallVector<mlir::Value, 4> compose(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange subscripts,
                                          llvm::ArrayRef<mlir::Operation*> hops, llvm::ArrayRef<mlir::Value> rootLbs) {
  llvm::SmallVector<mlir::Value, 4> subs;
  for (mlir::Value s : subscripts) subs.push_back(toIndex(b, loc, s));
  for (auto [h, hop] : llvm::enumerate(hops)) {
    mlir::Value const next = viewedArray(hop);
    if (auto section = mlir::dyn_cast<hlfir::DesignateOp>(hop)) {
      // ``lo + (u - 1) * st`` for the j-th triplet dim (a section starts at 1); a scalar subscript stays.
      llvm::SmallVector<mlir::Value, 4> mapped;
      mlir::OperandRange const ops = section.getIndices();
      unsigned cursor = 0;
      unsigned j = 0;
      for (bool const triplet : section.getIsTriplet()) {
        if (!triplet) {
          mapped.push_back(toIndex(b, loc, ops[cursor++]));
          continue;
        }
        mlir::Value const lo = toIndex(b, loc, ops[cursor]);
        mlir::Value const st = toIndex(b, loc, ops[cursor + 2]);
        cursor += 3;
        mlir::Value offset = b.create<mlir::arith::SubIOp>(loc, subs[j++], toIndex(b, loc, {}));
        if (constantOr1(st) != 1) offset = b.create<mlir::arith::MulIOp>(loc, offset, st);
        mapped.push_back(b.create<mlir::arith::AddIOp>(loc, lo, offset).getResult());
      }
      subs = std::move(mapped);
    } else if (!mlir::isa<fir::ConvertOp>(hop)) {
      llvm::SmallVector<mlir::Value, 4> const from = lowerBounds(hop->getResult(0));
      llvm::SmallVector<mlir::Value, 4> const to =
          h + 1 == hops.size() ? llvm::SmallVector<mlir::Value, 4>(rootLbs) : lowerBounds(next);
      for (size_t k = 0; k < subs.size(); ++k)
        subs[k] = rebase(b, loc, subs[k], from[k], k < to.size() ? to[k] : mlir::Value{});
    }
  }
  return subs;
}

}  // namespace

llvm::SmallVector<mlir::Value, 4> arrayLowerBounds(mlir::Value array) { return lowerBounds(array); }

std::optional<llvm::SmallVector<mlir::Operation*, 4>> traceArrayViews(mlir::Value view, hlfir::DeclareOp root) {
  return traceViews(view, [&](mlir::Value v) { return v.getDefiningOp() == root.getOperation(); });
}

std::optional<llvm::SmallVector<mlir::Value, 4>> subscriptsInRoot(mlir::OpBuilder& b, mlir::Location loc,
                                                                  mlir::Value view, mlir::ValueRange subscripts,
                                                                  hlfir::DeclareOp root) {
  auto hops = traceArrayViews(view, root);
  if (!hops || subscripts.size() != arrayRank(view)) return std::nullopt;
  return compose(b, loc, subscripts, *hops, lowerBounds(root.getResult(0)));
}

std::optional<llvm::SmallVector<mlir::Value, 4>> subscriptsThroughViews(mlir::OpBuilder& b, mlir::Location loc,
                                                                        mlir::Value view, mlir::ValueRange subscripts,
                                                                        mlir::Value root,
                                                                        llvm::ArrayRef<mlir::Value> rootLbs) {
  auto hops = traceViews(view, [&](mlir::Value v) { return v == root; });
  if (!hops || subscripts.size() != arrayRank(view)) return std::nullopt;
  return compose(b, loc, subscripts, *hops, rootLbs);
}

llvm::SmallVector<mlir::Value, 4> rebaseSubscripts(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange subscripts,
                                                   mlir::Value from, mlir::Value to) {
  llvm::SmallVector<mlir::Value, 4> const fromLbs = lowerBounds(from);
  llvm::SmallVector<mlir::Value, 4> const toLbs = lowerBounds(to);
  llvm::SmallVector<mlir::Value, 4> subs;
  for (auto [k, s] : llvm::enumerate(subscripts)) {
    mlir::Value const lbFrom = k < fromLbs.size() ? fromLbs[k] : mlir::Value{};
    mlir::Value const lbTo = k < toLbs.size() ? toLbs[k] : mlir::Value{};
    subs.push_back(rebase(b, loc, toIndex(b, loc, s), lbFrom, lbTo));
  }
  return subs;
}

}  // namespace hlfir_bridge
