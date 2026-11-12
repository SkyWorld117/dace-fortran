// Keep integer STOP calls while exposing their dead continuation to CFG structurization.
#include "flang/Optimizer/Dialect/FIROps.h"
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
    return "Keep integer STOP calls and normalize their unreachable terminator for CFG structurization";
  }
  void runOnOperation() override {
    getOperation().walk([](fir::UnreachableOp unreachable) {
      auto call = mlir::dyn_cast_or_null<fir::CallOp>(unreachable->getPrevNode());
      auto func = unreachable->getParentOfType<mlir::func::FuncOp>();
      if (!call || !func || func.getNumResults() != 0 || unreachable->getParentOp() != func) return;
      auto callee = call.getCallee();
      if (!callee || callee->getLeafReference() != "_FortranAStopStatement") return;
      // STOP still terminates in emit_integer_stop (host exit / device trap).
      // This return is therefore dead, but unlike fir.unreachable it gives
      // lift-cf-to-scf the common terminator it needs to retain the entire CFG.
      mlir::OpBuilder builder(unreachable);
      builder.create<mlir::func::ReturnOp>(unreachable.getLoc());
      unreachable.erase();
    });
  }
};
}  // namespace
std::unique_ptr<mlir::Pass> createNormalizeStopControlFlowPass() {
  return std::make_unique<NormalizeStopControlFlowPass>();
}
}  // namespace hlfir_bridge
