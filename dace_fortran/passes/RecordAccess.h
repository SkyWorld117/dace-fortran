// ============================================================================
// RecordAccess.h  --  where an element of a view of an array lies in the
// array itself.
// ============================================================================
// After inlining, an access to an array of records often goes through views
// of it: an inlined dummy declared over it with bounds of its own
// (``q_c(nf)`` over ``q(0:nf-1)``), a section passed as an argument
// (``call inner(q(2:3))``, then ``v(i)`` in ``inner``), boxes and conversions
// in between.  Flattening the array needs every access as subscripts of the
// array itself; these helpers trace the views and compose the subscripts, in
// one place for every pass that needs them.
// ============================================================================
#pragma once

#include <optional>

#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace hlfir_bridge {

/// The array ``op`` is a view of -- what a declaration or box declares or boxes, a conversion's operand, a section's
/// base -- or null when ``op`` is no view of an array (an element designate, a component, a slice, a load).
mlir::Value viewedArray(mlir::Operation* op);

/// The views from ``view`` back to ``root`` -- declarations over the array with bounds of their own, boxes of it
/// (``fir.embox`` / ``fir.rebox``, with or without a shift), conversions and sections (``hlfir.designate`` with
/// triplets or scalar subscripts) -- access first.  nullopt when ``view`` does not reach ``root`` through views alone,
/// or a declaration or box changes the rank (sequence association, not modelled).
std::optional<llvm::SmallVector<mlir::Operation*, 4>> traceArrayViews(mlir::Value view, hlfir::DeclareOp root);

/// The subscripts, in ``root``'s own index space, of the element that ``subscripts`` select in ``view``.  The views
/// of traceArrayViews compose root-ward: a declaration or box with lower bounds ``vlb`` over an array with lower bounds
/// ``alb`` maps a subscript ``u`` to ``alb + (u - vlb)``; a section ``lo:hi:st`` maps its j-th triplet subscript ``u``
/// to ``lo + (u - 1) * st`` and keeps its scalar subscripts.  The index arithmetic is emitted at ``b``'s insertion
/// point; nullopt, emitting nothing, when traceArrayViews finds no path.
std::optional<llvm::SmallVector<mlir::Value, 4>> subscriptsInRoot(mlir::OpBuilder& b, mlir::Location loc,
                                                                  mlir::Value view, mlir::ValueRange subscripts,
                                                                  hlfir::DeclareOp root);

/// ``subscripts`` of the leading dims of ``from`` as subscripts of the same elements of ``to``, which holds them in
/// the same positions with lower bounds of its own (a flat companion of a record array, say).
llvm::SmallVector<mlir::Value, 4> rebaseSubscripts(mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange subscripts,
                                                   mlir::Value from, mlir::Value to);

}  // namespace hlfir_bridge
