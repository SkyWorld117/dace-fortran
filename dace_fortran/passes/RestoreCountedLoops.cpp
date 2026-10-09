// ============================================================================
// RestoreCountedLoops.cpp  --  Turn flang's trip-counted ``scf.while`` back
// into the counted ``fir.do_loop`` of the DO it came from.
// ============================================================================
// Problem:
//     Flang lowers a DO loop whose body leaves structured control flow -- a
//     CYCLE inside an IF construct is enough -- as unstructured ``cf`` blocks
//     driven by a trip counter:
//         trip = (ub - lb + step) / step ; iv = lb
//         ^loop: if (trip > 0) { BODY ; trip = trip - 1 ; iv = iv + step ; br ^loop }
//     ``lift-cf-to-scf`` raises that to
//         scf.while : () -> () {
//           %t = fir.load %trip
//           %c = arith.cmpi sgt, %t, %c0
//           %r = scf.if %c { BODY ; trip = trip - 1 ; iv = iv + step ; yield 1 } else { yield 0 }
//           scf.condition(trunci %r)
//         } do { scf.yield }
//     (the branch taken comes back as an ``scf.if`` result; once canonicalised
//     the condition is ``%c`` itself), which the bridge extracts as a
//     ``while True`` with a break.  DaCe then
//     sees no iteration count: the loop is never turned into a map, and
//     neither is any loop nested in it -- the whole kernel, when the CYCLE
//     sits in the innermost loop of a cell-loop nest.
//
// What this pass does:
//     Matches exactly that shape -- the trip counter is a bare ``fir.alloca``
//     nothing else uses, the condition compares it with zero, the body ends
//     with the two updates -- and rewrites it to the counted loop flang emits
//     for a structured body:
//         fir.do_loop %i = lb to ub step step {
//           fir.store %i to %iv            ! the loop variable BODY reads
//           BODY
//         }
//         fir.store lb + max(trip, 0) * step to %iv  ! its value after the DO
//     ``ub`` is the one flang computed the trip count from when the count has
//     its ``(ub - lb + step) / step`` shape, and ``lb + (trip - 1) * step``
//     otherwise.  A CYCLE is by then an ``scf.if`` around the rest of the
//     body, which the structured loop keeps as it is.  A loop with an EXIT
//     tests an exit flag in its condition and does not match: it stays a
//     while loop.
//
// Why it's safe:
//     Only the expression of the iteration count changes.  The body, the
//     loop variable's value in every iteration and after the loop, and the
//     order of the iterations are those of the original DO.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// The pieces of one trip-counted while loop (see the file header).
struct TripCountedLoop {
  mlir::scf::WhileOp loop;
  mlir::scf::IfOp body;    // ``scf.if %c { BODY ; updates }``
  fir::AllocaOp trip;      // the trip counter
  fir::StoreOp tripInit;   // ``trip = count`` before the loop
  mlir::Value iv;          // the loop variable's memory
  fir::StoreOp ivInit;     // ``iv = lb`` before the loop
  mlir::Value step;        // what each iteration adds to the loop variable
  llvm::SmallVector<mlir::Operation*, 6> updates;  // the three ops of each update, to erase
};

/// The op before ``op`` in its block that is not a constant (flang materialises a constant next to each use).
mlir::Operation* prevNonConstant(mlir::Operation* op) {
  op = op->getPrevNode();
  while (op && mlir::isa<mlir::arith::ConstantOp>(op)) op = op->getPrevNode();
  return op;
}

/// Same SSA value, or two constants of the same integer value.
bool sameValue(mlir::Value a, mlir::Value b) {
  if (a == b) return true;
  llvm::APInt x, y;
  return mlir::matchPattern(a, mlir::m_ConstantInt(&x)) && mlir::matchPattern(b, mlir::m_ConstantInt(&y)) && x == y;
}

/// ``store (load %mem) <op> %delta to %mem``: the three ops ending at ``store`` with nothing but constants between
/// them.  Returns the delta and fills ``ops``; null when ``store`` does not end such a chain.
template <typename BinOp>
mlir::Value matchUpdate(fir::StoreOp store, llvm::SmallVectorImpl<mlir::Operation*>& ops) {
  auto bin = store.getValue().getDefiningOp<BinOp>();
  if (!bin || prevNonConstant(store) != bin.getOperation()) return {};
  auto load = bin.getLhs().template getDefiningOp<fir::LoadOp>();
  if (!load || prevNonConstant(bin) != load.getOperation() || load.getMemref() != store.getMemref()) return {};
  ops.push_back(load);
  ops.push_back(bin);
  ops.push_back(store);
  return bin.getRhs();
}

/// Does the loop go on exactly when ``body``'s then branch ran?  ``cond`` is either the comparison itself
/// (canonicalised IR) or the truncation of a result the two branches yield as different constants (raw
/// ``lift-cf-to-scf`` output, which encodes the branch taken in the if's results).
bool continuesAfterBody(mlir::Value cond, mlir::arith::CmpIOp cmp, mlir::scf::IfOp body) {
  if (cond == cmp.getResult()) return true;
  auto trunc = cond.getDefiningOp<mlir::arith::TruncIOp>();
  if (!trunc) return false;
  auto result = mlir::dyn_cast<mlir::OpResult>(trunc.getIn());
  if (!result || result.getOwner() != body.getOperation()) return false;
  unsigned const k = result.getResultNumber();
  auto thenYield = mlir::cast<mlir::scf::YieldOp>(body.getThenRegion().front().getTerminator());
  auto elseYield = mlir::cast<mlir::scf::YieldOp>(body.getElseRegion().front().getTerminator());
  llvm::APInt onThen, onElse;
  return mlir::matchPattern(thenYield.getOperand(k), mlir::m_ConstantInt(&onThen)) &&
         mlir::matchPattern(elseYield.getOperand(k), mlir::m_ConstantInt(&onElse)) && !onThen.isZero() &&
         onElse.isZero();
}

/// The loop ending at ``w`` when it has the trip-counted shape; nullopt otherwise.
std::optional<TripCountedLoop> matchTripCountedLoop(mlir::scf::WhileOp w) {
  if (w.getNumResults() != 0 || !w.getInits().empty()) return std::nullopt;
  // After region: nothing but the yield.
  auto& after = w.getAfter().front();
  if (after.getOperations().size() != 1 || !mlir::isa<mlir::scf::YieldOp>(after.front())) return std::nullopt;
  // Before region: load trip, compare with zero, the body under the comparison, the condition.
  llvm::SmallVector<mlir::Operation*, 6> ops;
  for (auto& op : w.getBefore().front())
    if (!mlir::isa<mlir::arith::ConstantOp>(op)) ops.push_back(&op);
  if (ops.size() < 4 || ops.size() > 5) return std::nullopt;
  auto load = mlir::dyn_cast<fir::LoadOp>(ops[0]);
  auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(ops[1]);
  auto body = mlir::dyn_cast<mlir::scf::IfOp>(ops[2]);
  auto cond = mlir::dyn_cast<mlir::scf::ConditionOp>(ops.back());
  if (!load || !cmp || !body || !cond) return std::nullopt;
  if (ops.size() == 5 && ops[3] != cond.getCondition().getDefiningOp()) return std::nullopt;
  if (cmp.getPredicate() != mlir::arith::CmpIPredicate::sgt || cmp.getLhs() != load.getResult() ||
      !mlir::matchPattern(cmp.getRhs(), mlir::m_Zero()))
    return std::nullopt;
  if (body.getCondition() != cmp.getResult() || !cond.getArgs().empty()) return std::nullopt;
  if (body.getNumResults() > 0 && body.getElseRegion().empty()) return std::nullopt;
  // The else branch runs once, when the loop ends; the rewrite has no place for anything it does.
  if (!body.getElseRegion().empty())
    for (auto& op : body.getElseRegion().front().without_terminator())
      if (!mlir::isa<mlir::arith::ConstantOp>(op)) return std::nullopt;
  if (!continuesAfterBody(cond.getCondition(), cmp, body)) return std::nullopt;
  auto trip = load.getMemref().getDefiningOp<fir::AllocaOp>();
  if (!trip) return std::nullopt;

  // The body ends with the two updates, in either order, then its yield.
  auto& then = body.getThenRegion().front();
  TripCountedLoop m{w, body, trip, {}, {}, {}, {}, {}};
  auto last = mlir::dyn_cast_or_null<fir::StoreOp>(prevNonConstant(then.getTerminator()));
  if (!last) return std::nullopt;
  llvm::SmallVector<mlir::Operation*, 3> lastOps;
  if (last.getMemref() == trip.getResult()) {
    mlir::Value delta = matchUpdate<mlir::arith::SubIOp>(last, lastOps);
    if (!delta || !mlir::matchPattern(delta, mlir::m_One())) return std::nullopt;
  } else {
    mlir::Value delta = matchUpdate<mlir::arith::AddIOp>(last, lastOps);
    if (!delta) return std::nullopt;
    m.iv = last.getMemref();
    m.step = delta;
  }
  auto first = mlir::dyn_cast_or_null<fir::StoreOp>(prevNonConstant(lastOps.front()));
  if (!first) return std::nullopt;
  llvm::SmallVector<mlir::Operation*, 3> firstOps;
  if (m.iv) {
    if (first.getMemref() != trip.getResult()) return std::nullopt;
    mlir::Value delta = matchUpdate<mlir::arith::SubIOp>(first, firstOps);
    if (!delta || !mlir::matchPattern(delta, mlir::m_One())) return std::nullopt;
  } else {
    mlir::Value delta = matchUpdate<mlir::arith::AddIOp>(first, firstOps);
    if (!delta) return std::nullopt;
    m.iv = first.getMemref();
    m.step = delta;
  }
  m.updates.append(firstOps);
  m.updates.append(lastOps);
  // The step is loop-invariant: defined outside the loop, or a constant.
  if (auto* def = m.step.getDefiningOp(); def && w->isAncestor(def) && !mlir::isa<mlir::arith::ConstantOp>(def))
    return std::nullopt;
  // The body moves into the counted loop: of the values the before region defines, it may use only constants
  // (cloned into the loop), not the trip counter's load or comparison.
  mlir::Block* beforeBlock = &w.getBefore().front();
  bool usesBeforeValue = false;
  then.walk([&](mlir::Operation* inner) {
    for (mlir::Value v : inner->getOperands())
      if (auto* def = v.getDefiningOp(); def && def->getBlock() == beforeBlock && !mlir::isa<mlir::arith::ConstantOp>(def))
        usesBeforeValue = true;
  });
  if (usesBeforeValue) return std::nullopt;

  // The trip counter: initialised right before the loop, otherwise touched only by the pattern.
  for (mlir::Operation* user : trip.getResult().getUsers()) {
    if (user == load.getOperation() || llvm::is_contained(m.updates, user)) continue;
    auto store = mlir::dyn_cast<fir::StoreOp>(user);
    if (!store || store->getBlock() != w->getBlock() || !store->isBeforeInBlock(w) || m.tripInit)
      return std::nullopt;
    m.tripInit = store;
  }
  if (!m.tripInit) return std::nullopt;
  // The loop variable: set right before the loop, not written in the body except by the update.
  bool ivWrittenInBody = false;
  body.getThenRegion().walk([&](fir::StoreOp store) {
    if (store.getMemref() == m.iv && !llvm::is_contained(m.updates, store.getOperation())) ivWrittenInBody = true;
  });
  if (ivWrittenInBody) return std::nullopt;
  // Flang stores the counter and the loop variable right before the loop; a store further up may belong to
  // another DO of the same variable.
  mlir::Operation* op = w->getPrevNode();
  for (int distance = 0; op && distance < 8; op = op->getPrevNode(), ++distance) {
    if (op->getNumRegions() > 0) return std::nullopt;
    if (auto store = mlir::dyn_cast<fir::StoreOp>(op); store && store.getMemref() == m.iv) {
      m.ivInit = store;
      break;
    }
  }
  if (!m.ivInit) return std::nullopt;
  return m;
}

/// ``ub`` when the trip count was computed as flang's ``(ub - lb + step) / step`` from this loop's ``lb`` and
/// ``step`` (the division is left out for a unit step once folded); null otherwise.
mlir::Value upperBoundOfTripCount(TripCountedLoop& m) {
  mlir::Value count = m.tripInit.getValue();
  if (auto div = count.getDefiningOp<mlir::arith::DivSIOp>()) {
    if (!sameValue(div.getRhs(), m.step)) return {};
    count = div.getLhs();
  }
  auto add = count.getDefiningOp<mlir::arith::AddIOp>();
  if (!add || !sameValue(add.getRhs(), m.step)) return {};
  auto sub = add.getLhs().getDefiningOp<mlir::arith::SubIOp>();
  if (!sub || !sameValue(sub.getRhs(), m.ivInit.getValue())) return {};
  return sub.getLhs();
}

void restore(TripCountedLoop& m) {
  mlir::OpBuilder b(m.loop);
  mlir::Location loc = m.loop.getLoc();
  auto idx = b.getIndexType();
  auto toIndex = [&](mlir::Value v) -> mlir::Value {
    return v.getType() == idx ? v : b.create<fir::ConvertOp>(loc, idx, v).getResult();
  };
  // A constant step may be defined in the loop body (flang materialises constants at their use).
  mlir::Value stepValue = m.step;
  if (auto* def = stepValue.getDefiningOp(); def && m.loop->isAncestor(def)) stepValue = b.clone(*def)->getResult(0);
  mlir::Value lb = toIndex(m.ivInit.getValue());
  mlir::Value step = toIndex(stepValue);
  mlir::Value trip = toIndex(m.tripInit.getValue());
  mlir::Value ub = upperBoundOfTripCount(m);
  if (ub) {
    ub = toIndex(ub);
  } else {
    auto one = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto lastIter = b.create<mlir::arith::SubIOp>(loc, trip, one);
    ub = b.create<mlir::arith::AddIOp>(loc, lb, b.create<mlir::arith::MulIOp>(loc, lastIter, step));
  }
  auto loop = b.create<fir::DoLoopOp>(loc, lb, ub, step, /*unordered=*/false, /*finalCountValue=*/false);
  mlir::Block& body = *loop.getBody();
  if (body.empty() || !body.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    mlir::OpBuilder tb(&body, body.end());
    tb.create<fir::ResultOp>(loc);
  }
  mlir::Type ivType = fir::unwrapRefType(m.iv.getType());
  {
    mlir::OpBuilder ib(&body, body.begin());
    mlir::Value i = loop.getInductionVar();
    if (ivType != idx) i = ib.create<fir::ConvertOp>(loc, ivType, i);
    ib.create<fir::StoreOp>(loc, i, m.iv);
  }
  for (mlir::Operation* op : llvm::reverse(m.updates)) op->erase();
  mlir::Block& then = m.body.getThenRegion().front();
  // Constants of the before region the body uses: cloned to the top of the counted loop.
  {
    mlir::OpBuilder cb(&body, std::next(body.begin()));
    mlir::Block* beforeBlock = &m.loop.getBefore().front();
    llvm::DenseMap<mlir::Value, mlir::Value> cloned;
    then.walk([&](mlir::Operation* inner) {
      for (mlir::OpOperand& use : inner->getOpOperands()) {
        auto* def = use.get().getDefiningOp();
        if (!def || def->getBlock() != beforeBlock) continue;
        auto [it, inserted] = cloned.try_emplace(use.get());
        if (inserted) it->second = cb.clone(*def)->getResult(0);
        use.set(it->second);
      }
    });
  }
  body.getOperations().splice(body.getTerminator()->getIterator(), then.getOperations(), then.begin(),
                              std::prev(then.end()));
  // The loop variable after the DO: lb plus a step per iteration run, lb when it ran none (a negative trip count).
  {
    mlir::OpBuilder ab(m.loop);
    ab.setInsertionPointAfter(loop);
    mlir::Value zero = ab.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value ran = ab.create<mlir::arith::MaxSIOp>(loc, trip, zero);
    mlir::Value final = ab.create<mlir::arith::AddIOp>(loc, lb, ab.create<mlir::arith::MulIOp>(loc, ran, step));
    if (ivType != idx) final = ab.create<fir::ConvertOp>(loc, ivType, final);
    ab.create<fir::StoreOp>(loc, final, m.iv);
  }
  m.loop.erase();
  m.tripInit.erase();
  if (m.trip.getResult().use_empty()) m.trip.erase();
}

struct RestoreCountedLoopsPass
    : public mlir::PassWrapper<RestoreCountedLoopsPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RestoreCountedLoopsPass)

  llvm::StringRef getArgument() const final { return "hlfir-restore-counted-loops"; }
  llvm::StringRef getDescription() const final {
    return "Rewrite the trip-counted scf.while that lift-cf-to-scf makes of a DO with a CYCLE "
           "back into a counted fir.do_loop.";
  }

  void runOnOperation() override {
    // Innermost first: an outer loop's body keeps the ops of the inner loop, restored or not.
    llvm::SmallVector<TripCountedLoop, 8> loops;
    getOperation().walk<mlir::WalkOrder::PostOrder>([&](mlir::scf::WhileOp w) {
      if (auto m = matchTripCountedLoop(w)) loops.push_back(*m);
    });
    for (auto& m : loops) restore(m);
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createRestoreCountedLoopsPass() { return std::make_unique<RestoreCountedLoopsPass>(); }

}  // namespace hlfir_bridge
