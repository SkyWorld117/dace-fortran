// Keep STOP calls while exposing their dead continuation to CFG structurization.
#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {
namespace {
struct NormalizeStopControlFlowPass
    : public mlir::PassWrapper<NormalizeStopControlFlowPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeStopControlFlowPass)
  llvm::StringRef getArgument() const final { return "hlfir-normalize-stop-control-flow"; }
  llvm::StringRef getDescription() const final {
    return "Keep integer STOP calls and normalize dead returns in subroutines and real functions";
  }
  void runOnOperation() override {
    getOperation().walk([](fir::UnreachableOp unreachable) {
      auto call = mlir::dyn_cast_or_null<fir::CallOp>(unreachable->getPrevNode());
      auto func = unreachable->getParentOfType<mlir::func::FuncOp>();
      if (!call || !func || unreachable->getParentOp() != func) return;
      if (func.getNumResults() > 1) return;
      if (func.getNumResults() == 1 && !mlir::isa<mlir::FloatType>(func.getResultTypes()[0])) return;
      auto callee = call.getCallee();
      if (!callee || callee->getLeafReference() != "_FortranAStopStatement") return;
      // STOP still terminates in emit_integer_stop (host exit / device trap).
      // This return is therefore dead, but unlike fir.unreachable it gives
      // lift-cf-to-scf the common terminator it needs to retain the entire CFG.
      mlir::OpBuilder builder(unreachable);
      // A scalar real helper still needs a typed return value on this dead
      // path. STOP executes first, so this zero can never enter the physics.
      // Keeping a common return lets the helper structurize and inline.
      if (func.getNumResults() == 1) {
        auto type = func.getResultTypes()[0];
        auto zero = builder.create<mlir::arith::ConstantOp>(
            unreachable.getLoc(), type, builder.getFloatAttr(type, 0.0));
        builder.create<mlir::func::ReturnOp>(unreachable.getLoc(), zero.getResult());
      } else {
        builder.create<mlir::func::ReturnOp>(unreachable.getLoc());
      }
      unreachable.erase();
    });
  }
};
}  // namespace
std::unique_ptr<mlir::Pass> createNormalizeStopControlFlowPass() {
  return std::make_unique<NormalizeStopControlFlowPass>();
}
}  // namespace hlfir_bridge
