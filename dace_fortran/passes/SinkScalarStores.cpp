// ============================================================================
// SinkScalarStores.cpp  --  move a scalar store below the uses of the value
// it overwrites, so a load reads what the memory it names still holds.
// ============================================================================
// Problem:
//     Flang fills an array constructor with an implied-do item
//     (``v = [-1d0, (i*a, i=1,3)]``) through inline element stores at
//     positions kept in an anonymous counter, and advances the counter
//     BEFORE the element store that uses its old value:
//
//       %p  = fir.load %pos
//       %p1 = arith.addi %p, %c1
//       fir.store %p1 to %pos
//       %e  = hlfir.designate %ctor (%p)
//       hlfir.assign %x to %e
//
//     The bridge renders a load as the name of the memory it reads, so the
//     subscript ``%p`` would name the advanced counter and every element
//     would land one position later.
//
// Fix:
//     For a scalar ``fir.alloca`` whose address never escapes (every user,
//     through ``hlfir.declare`` / ``fir.convert``, is a load, a store into
//     it, or a scalar ``hlfir.assign`` into it), move each write below the
//     last operation of its block that uses a value computed from an earlier
//     load of the same memory in that block, directly or through what other
//     operations compute from it (the element address above):
//
//       %p  = fir.load %pos
//       %p1 = arith.addi %p, %c1
//       %e  = hlfir.designate %ctor (%p)
//       hlfir.assign %x to %e
//       fir.store %p1 to %pos
//
// Safety:
//     A write moves only when no other access to the memory lies between its
//     old and its new position; with an address that never escapes, nothing
//     else can reach the memory, so every access sees the value it saw
//     before.  A write whose old value is still used by the block terminator
//     or outside the block stays where it is (the bridge then refuses the
//     stale subscript instead of reading the advanced value).
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

bool isScalar(mlir::Type t) {
  return mlir::isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType, fir::LogicalType>(t);
}

/// Collect every load of and write into the memory ``v`` designates, through ``hlfir.declare`` / ``fir.convert``.
/// False when the address reaches anything else (a call, a stored pointer, a box, ...).
bool collectAccesses(mlir::Value v, llvm::SmallVectorImpl<fir::LoadOp>& loads,
                     llvm::SmallVectorImpl<mlir::Operation*>& writes) {
  for (mlir::OpOperand& use : v.getUses()) {
    mlir::Operation* u = use.getOwner();
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
      if (use.get() != decl.getMemref()) return false;
      for (mlir::Value r : decl.getResults())
        if (!collectAccesses(r, loads, writes)) return false;
      continue;
    }
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(u)) {
      if (!collectAccesses(conv.getResult(), loads, writes)) return false;
      continue;
    }
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(u)) {
      loads.push_back(ld);
      continue;
    }
    if (auto st = mlir::dyn_cast<fir::StoreOp>(u); st && st.getValue() != use.get()) {
      writes.push_back(st);
      continue;
    }
    if (auto as = mlir::dyn_cast<hlfir::AssignOp>(u);
        as && as.getRhs() != use.get() && isScalar(as.getRhs().getType())) {
      writes.push_back(as);
      continue;
    }
    return false;
  }
  return true;
}

/// The last operation of ``write``'s block that uses a value computed from one of ``loads`` ahead of ``write`` in
/// that block -- directly or through what other operations compute from it (the element address a designate forms
/// from the old position, which the element store then uses): the point ``write`` has to move below.  nullptr when
/// no such value is used after ``write``; ``write`` itself when one is used where ``write`` cannot follow (the block
/// terminator, another block).
mlir::Operation* lastUseOfOverwrittenValue(mlir::Operation* write, llvm::ArrayRef<fir::LoadOp> loads) {
  mlir::Block* block = write->getBlock();
  llvm::SmallVector<mlir::Value> work;
  for (fir::LoadOp ld : loads)
    if (ld->getBlock() == block && ld->isBeforeInBlock(write)) work.push_back(ld.getResult());
  llvm::DenseSet<mlir::Operation*> seen;
  mlir::Operation* last = nullptr;
  while (!work.empty()) {
    mlir::Value v = work.pop_back_val();
    for (mlir::Operation* user : v.getUsers()) {
      if (user == write) continue;
      mlir::Operation* top = block->findAncestorOpInBlock(*user);
      if (!top || top->hasTrait<mlir::OpTrait::IsTerminator>()) return write;
      if (write->isBeforeInBlock(top) && (!last || last->isBeforeInBlock(top))) last = top;
      if (seen.insert(top).second)
        for (mlir::Value r : top->getResults()) work.push_back(r);
    }
  }
  return last;
}

struct SinkScalarStoresPass : public mlir::PassWrapper<SinkScalarStoresPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkScalarStoresPass)

  llvm::StringRef getArgument() const final { return "hlfir-sink-scalar-stores"; }
  llvm::StringRef getDescription() const final {
    return "Move a store into a scalar whose address never escapes below the uses of the value it overwrites.";
  }

  void runOnOperation() override {
    // Collected first: moving a write while walking could skip the operations it moves past.
    llvm::SmallVector<fir::AllocaOp> allocas;
    getOperation().walk([&](fir::AllocaOp alloca) {
      if (isScalar(alloca.getAllocatedType())) allocas.push_back(alloca);
    });
    for (fir::AllocaOp alloca : allocas) {
      llvm::SmallVector<fir::LoadOp> loads;
      llvm::SmallVector<mlir::Operation*> writes;
      if (!collectAccesses(alloca.getResult(), loads, writes) || loads.empty()) continue;
      for (mlir::Operation* write : writes) {
        mlir::Operation* last = lastUseOfOverwrittenValue(write, loads);
        if (!last || last == write) continue;
        mlir::Block* block = write->getBlock();
        auto inRange = [&](mlir::Operation* op) {
          mlir::Operation* top = block->findAncestorOpInBlock(*op);
          return top && write->isBeforeInBlock(top) && !last->isBeforeInBlock(top);
        };
        if (llvm::any_of(loads, [&](fir::LoadOp ld) { return inRange(ld); }) || llvm::any_of(writes, inRange)) continue;
        write->moveAfter(last);
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createSinkScalarStoresPass() { return std::make_unique<SinkScalarStoresPass>(); }

}  // namespace hlfir_bridge
