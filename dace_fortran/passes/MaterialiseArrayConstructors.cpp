// ============================================================================
// MaterialiseArrayConstructors.cpp  --  build every array constructor in a
// named local array the bridge can see.
// ============================================================================
//
// Flang builds an array constructor in a heap temp named ``.tmp.arrayctor``
// and hands the value on with ``hlfir.as_expr %temp move %true``.  It fills
// the temp in one of two ways:
//
// * Inline stores, when every item is a scalar (``b = [x, q(j, i), q(j, i + 1)]``)
//   or an implied-do of known extent:
//
//     %t    = fir.allocmem !fir.array<NxT> {bindc_name = ".tmp.arrayctor"}
//     %td:2 = hlfir.declare %t(%shape) {uniq_name = ".tmp.arrayctor"}
//     hlfir.assign %v1 to (hlfir.designate %td#0 (%c1))
//     ...
//     %e    = hlfir.as_expr %td#0 move %true
//     hlfir.assign %e to %dst
//
// * Through the Fortran runtime, when an item is array-valued
//   (``u = [s, s*w(1:3), z(2:3), t]``):
//
//     %bx   = fir.embox %td#1(%shape)
//     fir.store %bx to %boxref
//     fir.call @_FortranAInitArrayConstructorVector(%state, %boxref', ...)
//     fir.store %v to %scratch                       // scalar item
//     fir.call @_FortranAPushArrayConstructorSimpleScalar(%state, %scratch')
//     %b    = fir.embox %var(%vshape)                // array item
//     fir.call @_FortranAPushArrayConstructorValue(%state, %b')
//
// The bridge cannot model either temp: the runtime calls are opaque (the
// assignment then reads a buffer nothing visibly writes, and its operand
// renders as ``?``), and an inline-store temp was only special-cased by
// retargeting its element stores at ``dst`` -- without their reads, at the
// wrong elements of a section ``dst`` and wrongly when an item reads ``dst``.
//
// This pass gives both forms a named local array instead
// (static constructor extent N, trivial element type):
//
//     %a    = fir.alloca !fir.array<NxT>
//     %ad:2 = hlfir.declare %a(%shape) {uniq_name = "_QF<encl>Earray_ctor_<n>"}
//     <element stores into %ad>
//     hlfir.assign %ad#0 to %dst
//
// An inline-store temp keeps its element stores, now into ``%ad``.  The
// runtime protocol is replaced, at each push, by one assignment per scalar
// item and one loop nest per array item (array element order):
//
//     fir.do_loop %k = 1 to extent {
//       %x = fir.load (hlfir.designate %var (%k + lb - 1))
//       hlfir.assign %x to (hlfir.designate %ad#0 (pos + %k - 1))
//     }
//
// An array item's elements are read through ``hlfir.designate`` of the
// pushed variable, the element-of-section form flang emits inside an
// ``hlfir.elemental``, so record members, sections and lower bounds resolve
// through the bridge's existing designate handling.  Consumers of the value
// read ``%ad``: the assignment copies it, ``hlfir.apply`` becomes an element
// load, ``hlfir.destroy`` goes.  Because the temp is a real array, an item
// that reads ``dst`` itself (``w = [w(3:4), w(1:2)]``) still sees the old
// values.
//
// Runs after ``hlfir-expand-vector-subscript-gather``, which has already
// materialised every elemental item (``s*w(1:3)``) into a named local array,
// so array items are always variables here: an unsliced ``fir.embox`` of a
// variable, or a descriptor-carrying variable pushed as it is (a strided
// section, an assumed-shape dummy; extents then come from its shape or from
// ``fir.box_dims``).
//
// A constructor outside this scope (runtime extent, runtime pushes from an
// implied-do, derived-type or character elements, a sliced embox or a rebox
// as item, an ``hlfir.as_expr`` consumer other than an assignment,
// ``hlfir.apply`` or ``hlfir.destroy``) is left untouched.
// ============================================================================

#include <optional>
#include <string>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

constexpr llvm::StringLiteral kInit = "_FortranAInitArrayConstructorVector";
constexpr llvm::StringLiteral kPushScalar = "_FortranAPushArrayConstructorSimpleScalar";
constexpr llvm::StringLiteral kPushValue = "_FortranAPushArrayConstructorValue";

bool calls(fir::CallOp call, llvm::StringRef name) {
  auto sym = call.getCallee();
  return sym && sym->getLeafReference().getValue() == name;
}

mlir::Value peelConvert(mlir::Value v) {
  while (auto cv = v.getDefiningOp<fir::ConvertOp>()) v = cv.getValue();
  return v;
}

std::optional<int64_t> constIndex(mlir::Value v) {
  if (auto c = v.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto a = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) return a.getInt();
  return std::nullopt;
}

/// One item pushed onto the constructor, in push order.
struct Item {
  fir::CallOp push;
  mlir::Value scalar;                         // scalar item: the value stored into the push's scratch slot
  mlir::Value var;                            // array item: the pushed variable (designate / declare result)
  mlir::Type eleTy;                           // array item: the variable's element type
  unsigned rank = 0;                          // array item: 0 for a boxed scalar
  llvm::SmallVector<mlir::Value, 2> extents;  // empty with rank > 0: read from the box with fir.box_dims
  llvm::SmallVector<mlir::Value, 2> lbounds;  // empty: every dim starts at 1
};

/// The constructor an ``_FortranAInitArrayConstructorVector`` call starts, once validated.
struct Ctor {
  fir::CallOp init;
  fir::StoreOp boxStore;
  fir::EmboxOp tmpBox;
  hlfir::DeclareOp tmpDecl;
  fir::AllocMemOp tmpAlloc;
  fir::SequenceType seqTy;
  llvm::SmallVector<Item, 8> items;
  llvm::SmallVector<hlfir::AsExprOp, 2> exprs;
};

std::string enclosingName(mlir::Operation* op) {
  auto fn = op->getParentOfType<mlir::func::FuncOp>();
  if (!fn) return "anon";
  auto sym = fn.getSymName().str();
  if (sym.rfind("_QP", 0) == 0) return sym.substr(3);
  if (sym.rfind("_QM", 0) == 0) {
    auto p = sym.find('P', 3);
    if (p != std::string::npos) return sym.substr(3, p - 3) + "_" + sym.substr(p + 1);
    return sym.substr(3);
  }
  return sym;
}

/// The value stored into ``slot`` last before ``push`` (scanning back no further than ``init``).
mlir::Value storedBefore(fir::CallOp push, fir::CallOp init, mlir::Value slot) {
  for (mlir::Operation* op = push->getPrevNode(); op && op != init.getOperation(); op = op->getPrevNode())
    if (auto st = mlir::dyn_cast<fir::StoreOp>(op); st && st.getMemref() == slot) return st.getValue();
  return {};
}

/// The heap temp flang declares for an array constructor.
bool isCtorTemp(hlfir::DeclareOp decl) { return decl.getUniqName().str().find(".tmp.arrayctor") != std::string::npos; }

/// The consumers of a constructor's value this pass can point at the local temp.
bool supportedConsumers(hlfir::AsExprOp ae) {
  for (mlir::Operation* u : ae.getResult().getUsers()) {
    if (auto as = mlir::dyn_cast<hlfir::AssignOp>(u); as && as.getRhs() == ae.getResult()) continue;
    if (mlir::isa<hlfir::DestroyOp>(u)) continue;
    if (auto ap = mlir::dyn_cast<hlfir::ApplyOp>(u); ap && ap.getTypeparams().empty()) continue;
    return false;
  }
  return true;
}

/// The static rank-1 trivial-element type of a constructor's heap temp, or null.
fir::SequenceType tempType(fir::AllocMemOp alloc) {
  auto seqTy = mlir::dyn_cast<fir::SequenceType>(alloc.getInType());
  if (!seqTy || seqTy.getDimension() != 1 || seqTy.hasDynamicExtents() || !fir::isa_trivial(seqTy.getEleTy()))
    return {};
  return seqTy;
}

/// A named local array in place of the heap temp ``tmpDecl`` declares (where it is declared: its shape dominates
/// there).
hlfir::DeclareOp makeLocalTemp(hlfir::DeclareOp tmpDecl, fir::AllocMemOp tmpAlloc, fir::SequenceType seqTy,
                               unsigned id) {
  mlir::OpBuilder b(tmpDecl);
  auto alloca = b.create<fir::AllocaOp>(tmpAlloc.getLoc(), seqTy);
  std::string const uniqName = "_QF" + enclosingName(tmpDecl) + "Earray_ctor_" + std::to_string(id);
  return b.create<hlfir::DeclareOp>(tmpDecl.getLoc(), alloca.getResult(), uniqName, tmpDecl.getShape());
}

/// Point the consumers of the constructor's value at the local temp and drop the expressions.
void readFromTemp(llvm::ArrayRef<hlfir::AsExprOp> exprs, mlir::Value temp, mlir::Type eleTy) {
  mlir::Type const eleRefTy = fir::ReferenceType::get(eleTy);
  for (auto ae : exprs) {
    for (mlir::Operation* u : llvm::make_early_inc_range(ae.getResult().getUsers())) {
      if (auto as = mlir::dyn_cast<hlfir::AssignOp>(u)) {
        as->setOperand(0, temp);
      } else if (auto ap = mlir::dyn_cast<hlfir::ApplyOp>(u)) {
        mlir::OpBuilder b(ap);
        auto el = b.create<hlfir::DesignateOp>(ap.getLoc(), eleRefTy, temp, ap.getIndices());
        auto ld = b.create<fir::LoadOp>(ap.getLoc(), el.getResult());
        ap.getResult().replaceAllUsesWith(ld.getResult());
        ap.erase();
      } else {
        u->erase();  // hlfir.destroy
      }
    }
    ae.erase();
  }
}

/// Erase the heap temp once nothing but ``fir.freemem`` uses it.
void eraseHeapTemp(hlfir::DeclareOp tmpDecl, fir::AllocMemOp tmpAlloc) {
  for (mlir::Value r : tmpDecl->getResults())
    for (mlir::Operation* u : llvm::make_early_inc_range(r.getUsers())) u->erase();  // fir.freemem
  tmpDecl.erase();
  for (mlir::Operation* u : llvm::make_early_inc_range(tmpAlloc->getUsers())) u->erase();
  tmpAlloc.erase();
}

/// A constructor filled by inline element stores: its temp is only designated, handed on and freed.
struct InlineCtor {
  hlfir::DeclareOp tmpDecl;
  fir::AllocMemOp tmpAlloc;
  fir::SequenceType seqTy;
  llvm::SmallVector<hlfir::AsExprOp, 2> exprs;
};

std::optional<InlineCtor> analyseInline(hlfir::DeclareOp decl) {
  if (!isCtorTemp(decl) || !decl.getShape()) return std::nullopt;
  InlineCtor c;
  c.tmpDecl = decl;
  c.tmpAlloc = decl.getMemref().getDefiningOp<fir::AllocMemOp>();
  if (!c.tmpAlloc) return std::nullopt;
  c.seqTy = tempType(c.tmpAlloc);
  if (!c.seqTy) return std::nullopt;
  for (mlir::Value r : decl->getResults())
    for (mlir::OpOperand& use : r.getUses()) {
      mlir::Operation* u = use.getOwner();
      if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u); dg && use.getOperandNumber() == 0) continue;
      if (auto ae = mlir::dyn_cast<hlfir::AsExprOp>(u)) {
        if (!supportedConsumers(ae)) return std::nullopt;
        c.exprs.push_back(ae);
        continue;
      }
      if (mlir::isa<fir::FreeMemOp>(u)) continue;
      return std::nullopt;  // e.g. the runtime protocol's descriptor
    }
  for (mlir::Operation* u : c.tmpAlloc->getUsers())
    if (u != decl.getOperation() && !mlir::isa<fir::FreeMemOp>(u)) return std::nullopt;
  return c;
}

mlir::Type peelRefBox(mlir::Type t) {
  for (;;) {
    if (auto r = mlir::dyn_cast<fir::ReferenceType>(t))
      t = r.getEleTy();
    else if (auto b = mlir::dyn_cast<fir::BoxType>(t))
      t = b.getEleTy();
    else if (auto h = mlir::dyn_cast<fir::HeapType>(t))
      t = h.getEleTy();
    else if (auto p = mlir::dyn_cast<fir::PointerType>(t))
      t = p.getEleTy();
    else
      return t;
  }
}

/// The lower bounds a declare gives its variable (empty: all 1).
llvm::SmallVector<mlir::Value, 2> declaredLowerBounds(hlfir::DeclareOp decl) {
  llvm::SmallVector<mlir::Value, 2> lbs;
  if (mlir::Value shape = decl.getShape()) {
    if (auto ss = shape.getDefiningOp<fir::ShapeShiftOp>()) {
      auto origins = ss.getOrigins();
      lbs.assign(origins.begin(), origins.end());
    } else if (auto sh = shape.getDefiningOp<fir::ShiftOp>()) {
      auto origins = sh.getOrigins();
      lbs.assign(origins.begin(), origins.end());
    }
  }
  return lbs;
}

bool extentsOf(mlir::Value shape, llvm::SmallVector<mlir::Value, 2>& out) {
  if (!shape) return false;
  if (auto sh = shape.getDefiningOp<fir::ShapeOp>()) {
    auto extents = sh.getExtents();
    out.assign(extents.begin(), extents.end());
    return true;
  }
  if (auto ss = shape.getDefiningOp<fir::ShapeShiftOp>()) {
    auto extents = ss.getExtents();
    out.assign(extents.begin(), extents.end());
    return true;
  }
  return false;
}

/// Validate an array (or boxed scalar) item: an unsliced ``fir.embox`` of an HLFIR variable, or a box-typed HLFIR
/// variable pushed as it is (a strided section, an assumed-shape dummy).
bool analyseValueItem(Item& item) {
  mlir::Value const pushed = peelConvert(item.push.getArgOperands()[1]);
  mlir::Value var = pushed;
  mlir::Value shape;
  auto box = pushed.getDefiningOp<fir::EmboxOp>();
  if (box) {
    if (box.getSlice() || !box.getTypeparams().empty()) return false;
    var = box.getMemref();
    shape = box.getShape();
  }
  mlir::Operation* def = var.getDefiningOp();
  if (!def) return false;
  if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
    var = decl.getResult(0);  // the variable, not the raw base, so designate indices use its bounds
    item.lbounds = declaredLowerBounds(decl);
    if (!shape) {
      mlir::Value const declShape = decl.getShape();
      if (declShape && mlir::isa<fir::ShapeOp, fir::ShapeShiftOp>(declShape.getDefiningOp())) shape = declShape;
    }
  } else if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
    if (!shape) shape = dg.getShape();  // a section starts at 1
  } else {
    return false;
  }
  // Pushed without an embox: only a descriptor-carrying variable is an array value as it stands.
  if (!box && !mlir::isa<fir::BoxType>(var.getType())) return false;
  mlir::Type const ty = peelRefBox(var.getType());
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(ty)) {
    if (!fir::isa_trivial(seq.getEleTy())) return false;
    item.eleTy = seq.getEleTy();
    item.rank = seq.getDimension();
    if (shape) {
      if (!extentsOf(shape, item.extents) || item.extents.size() != item.rank) return false;
    } else if (!mlir::isa<fir::BoxType>(var.getType())) {
      return false;  // no shape and no descriptor to read the extents from
    }
    if (!item.lbounds.empty() && item.lbounds.size() != item.rank) return false;
  } else if (fir::isa_trivial(ty)) {
    item.eleTy = ty;
    item.rank = 0;
  } else {
    return false;
  }
  item.var = var;
  return true;
}

std::optional<Ctor> analyse(fir::CallOp init) {
  Ctor c;
  c.init = init;
  if (init.getArgOperands().size() < 2) return std::nullopt;
  mlir::Value const state = init.getArgOperands()[0];
  mlir::Value const boxRef = peelConvert(init.getArgOperands()[1]);

  // The temp: the box stored into the descriptor slot just before the init call.
  for (mlir::Operation* op = init->getPrevNode(); op; op = op->getPrevNode())
    if (auto st = mlir::dyn_cast<fir::StoreOp>(op); st && st.getMemref() == boxRef) {
      c.boxStore = st;
      break;
    }
  if (!c.boxStore) return std::nullopt;
  c.tmpBox = c.boxStore.getValue().getDefiningOp<fir::EmboxOp>();
  if (!c.tmpBox || c.tmpBox.getSlice()) return std::nullopt;
  c.tmpDecl = c.tmpBox.getMemref().getDefiningOp<hlfir::DeclareOp>();
  if (!c.tmpDecl || !c.tmpDecl.getShape()) return std::nullopt;
  c.tmpAlloc = c.tmpDecl.getMemref().getDefiningOp<fir::AllocMemOp>();
  if (!c.tmpAlloc) return std::nullopt;
  c.seqTy = tempType(c.tmpAlloc);
  if (!c.seqTy) return std::nullopt;
  // The runtime may reallocate a descriptor it can read back; a load of the slot means the result is taken from it.
  for (mlir::Operation* u : boxRef.getUsers())
    if (mlir::isa<fir::LoadOp>(u)) return std::nullopt;

  // The pushes: calls on the same state after the init, up to the next constructor started on that state.
  for (mlir::Operation* op = init->getNextNode(); op; op = op->getNextNode()) {
    if (auto call = mlir::dyn_cast<fir::CallOp>(op)) {
      bool const onState = !call.getArgOperands().empty() && call.getArgOperands()[0] == state;
      if (onState && calls(call, kInit)) break;
      if (onState && (calls(call, kPushScalar) || calls(call, kPushValue))) {
        if (call.getArgOperands().size() < 2) return std::nullopt;
        Item item;
        item.push = call;
        if (calls(call, kPushScalar)) {
          item.scalar = storedBefore(call, init, peelConvert(call.getArgOperands()[1]));
          if (!item.scalar || !fir::isa_trivial(item.scalar.getType())) return std::nullopt;
        } else if (!analyseValueItem(item)) {
          return std::nullopt;
        }
        c.items.push_back(std::move(item));
        continue;
      }
      if (onState) return std::nullopt;
    }
    // A push inside a nested region (implied-do) is out of scope.
    if (op->getNumRegions() > 0) {
      bool nested = false;
      op->walk([&](fir::CallOp inner) {
        if (!inner.getArgOperands().empty() && inner.getArgOperands()[0] == state) nested = true;
      });
      if (nested) return std::nullopt;
    }
  }
  if (c.items.empty()) return std::nullopt;

  // Uses of the temp: its descriptor and the expression(s) taking ownership of it.
  for (mlir::Value r : c.tmpDecl->getResults())
    for (mlir::Operation* u : r.getUsers()) {
      if (u == c.tmpBox.getOperation()) continue;
      if (auto ae = mlir::dyn_cast<hlfir::AsExprOp>(u)) {
        if (!supportedConsumers(ae)) return std::nullopt;
        c.exprs.push_back(ae);
        continue;
      }
      if (mlir::isa<fir::FreeMemOp>(u)) continue;
      return std::nullopt;
    }
  for (mlir::Operation* u : c.tmpBox->getUsers())
    if (u != c.boxStore.getOperation()) return std::nullopt;
  for (mlir::Operation* u : c.tmpAlloc->getUsers())
    if (u != c.tmpDecl.getOperation() && !mlir::isa<fir::FreeMemOp>(u)) return std::nullopt;
  return c;
}

struct MaterialiseArrayConstructorsPass
    : public mlir::PassWrapper<MaterialiseArrayConstructorsPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterialiseArrayConstructorsPass)

  llvm::StringRef getArgument() const final { return "hlfir-materialise-array-constructors"; }
  llvm::StringRef getDescription() const final {
    return "Build each array constructor in a named local array: keep inline element stores, replace the Fortran "
           "runtime protocol (_FortranAInitArrayConstructorVector / _FortranAPushArrayConstructor*) with explicit "
           "element assignments, and point the constructor's consumers at the local array.";
  }

  void runOnOperation() override {
    unsigned counter = 0;
    llvm::SmallVector<fir::CallOp, 8> inits;
    getOperation().walk([&](fir::CallOp call) {
      if (calls(call, kInit)) inits.push_back(call);
    });
    llvm::SetVector<mlir::Operation*> slots;  // scratch / descriptor / state allocas to sweep
    for (auto init : inits)
      if (auto c = analyse(init)) rewrite(*c, counter++, slots);
    sweep(slots);

    llvm::SmallVector<hlfir::DeclareOp, 8> temps;
    getOperation().walk([&](hlfir::DeclareOp decl) {
      if (isCtorTemp(decl)) temps.push_back(decl);
    });
    for (auto decl : temps)
      if (auto c = analyseInline(decl)) rewriteInline(*c, counter++);
  }

 private:
  static void rewrite(Ctor& c, unsigned id, llvm::SetVector<mlir::Operation*>& slots) {
    auto loc = c.init.getLoc();
    mlir::Type const eleTy = c.seqTy.getEleTy();
    mlir::Type const eleRefTy = fir::ReferenceType::get(eleTy);

    // 1) The named local temp.
    mlir::Value const temp = makeLocalTemp(c.tmpDecl, c.tmpAlloc, c.seqTy, id).getResult(0);
    mlir::OpBuilder b(c.init);

    auto index = [&](int64_t v) -> mlir::Value {
      return b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(v)).getResult();
    };
    // pos (1-based) = constPos + dynPos; dynPos only when an item's extent is not a constant.
    int64_t constPos = 1;
    mlir::Value dynPos;
    auto positionPlus = [&](mlir::Value offset) -> mlir::Value {  // pos + offset, offset may be null
      mlir::Value p = index(constPos);
      if (dynPos) p = b.create<mlir::arith::AddIOp>(loc, dynPos, p);
      if (offset) p = b.create<mlir::arith::AddIOp>(loc, p, offset);
      return p;
    };
    auto store = [&](mlir::Value value, mlir::Value at) {
      if (value.getType() != eleTy) value = b.create<fir::ConvertOp>(loc, eleTy, value);
      auto dst = b.create<hlfir::DesignateOp>(loc, eleRefTy, temp, mlir::ValueRange{at});
      b.create<hlfir::AssignOp>(loc, value, dst.getResult());
    };

    // 2) One assignment per scalar item, one loop nest per array item, each where its push was.
    for (auto& item : c.items) {
      b.setInsertionPoint(item.push);
      if (item.scalar || item.rank == 0) {
        mlir::Value v = item.scalar ? item.scalar : b.create<fir::LoadOp>(loc, item.var).getResult();
        store(v, positionPlus({}));
        ++constPos;
        continue;
      }
      unsigned const rank = item.rank;
      mlir::Value const c1 = index(1);
      llvm::SmallVector<mlir::Value, 2> extents;
      if (item.extents.empty()) {
        for (unsigned d = 0; d < rank; ++d) {
          mlir::Type const idx = b.getIndexType();
          auto dims = b.create<fir::BoxDimsOp>(loc, idx, idx, idx, item.var, index(d));
          extents.push_back(dims.getResult(1));
        }
      } else {
        for (mlir::Value e : item.extents)
          extents.push_back(e.getType().isIndex() ? e : b.create<fir::ConvertOp>(loc, b.getIndexType(), e).getResult());
      }
      {
        // Column-major order: dim 0 varies fastest.  Loops nest outermost = last dim.
        mlir::OpBuilder::InsertionGuard guard(b);
        llvm::SmallVector<mlir::Value, 2> ivs(rank);
        for (int d = static_cast<int>(rank) - 1; d >= 0; --d) {
          auto loop = b.create<fir::DoLoopOp>(loc, c1, extents[d], c1, /*unordered=*/false, /*finalCountValue=*/false);
          ivs[d] = loop.getInductionVar();
          b.setInsertionPointToStart(loop.getBody());
        }
        llvm::SmallVector<mlir::Value, 2> srcIdx;
        for (unsigned d = 0; d < rank; ++d) {
          mlir::Value i = ivs[d];
          if (!item.lbounds.empty()) {
            mlir::Value lb = item.lbounds[d];
            if (!lb.getType().isIndex()) lb = b.create<fir::ConvertOp>(loc, b.getIndexType(), lb);
            if (constIndex(lb) != std::optional<int64_t>(1))
              i = b.create<mlir::arith::SubIOp>(loc, b.create<mlir::arith::AddIOp>(loc, i, lb), c1);
          }
          srcIdx.push_back(i);
        }
        auto src =
            b.create<hlfir::DesignateOp>(loc, fir::ReferenceType::get(item.eleTy), item.var, mlir::ValueRange{srcIdx});
        mlir::Value const v = b.create<fir::LoadOp>(loc, src.getResult());
        // Linear offset (0-based) = sum_d (iv_d - 1) * stride_d; stride_0 = 1, stride_d = stride_{d-1} * extent_{d-1}.
        mlir::Value lin;
        mlir::Value stride;
        for (unsigned d = 0; d < rank; ++d) {
          mlir::Value term = b.create<mlir::arith::SubIOp>(loc, ivs[d], c1);
          if (stride) term = b.create<mlir::arith::MulIOp>(loc, term, stride);
          lin = lin ? b.create<mlir::arith::AddIOp>(loc, lin, term).getResult() : term;
          stride = stride ? b.create<mlir::arith::MulIOp>(loc, stride, extents[d]).getResult() : extents[d];
        }
        store(v, positionPlus(lin));
      }
      // Advance the position past the item (at the push, outside the loops).
      int64_t constTotal = 1;
      bool allConst = true;
      for (mlir::Value e : extents) {
        if (auto ce = constIndex(e))
          constTotal *= *ce;
        else
          allConst = false;
      }
      if (allConst) {
        constPos += constTotal;
      } else {
        mlir::Value total = extents[0];
        for (unsigned d = 1; d < rank; ++d) total = b.create<mlir::arith::MulIOp>(loc, total, extents[d]);
        dynPos = dynPos ? b.create<mlir::arith::AddIOp>(loc, dynPos, total).getResult() : total;
      }
    }

    // 3) Consumers of the constructor's value read the local temp instead.
    readFromTemp(c.exprs, temp, eleTy);

    // 4) Drop the runtime protocol and the heap temp.
    // Ops feeding only the calls about to go, collected while the calls still hold them.
    llvm::SmallVector<mlir::Operation*, 16> work;
    for (auto& item : c.items) {
      for (mlir::Value a : item.push.getArgOperands())
        if (mlir::Operation* d = a.getDefiningOp()) work.push_back(d);
      item.push.erase();
    }
    for (mlir::Value a : c.init.getArgOperands())
      if (mlir::Operation* d = a.getDefiningOp()) work.push_back(d);
    c.init.erase();
    c.boxStore.erase();
    c.tmpBox.erase();
    eraseHeapTemp(c.tmpDecl, c.tmpAlloc);
    // Converts / embox / address_of left without users by the erased calls; remember the slots they addressed.
    llvm::DenseSet<mlir::Operation*> erased;
    while (!work.empty()) {
      mlir::Operation* op = work.pop_back_val();
      if (erased.contains(op)) continue;
      if (mlir::isa<fir::AllocaOp>(op)) {
        slots.insert(op);
        continue;
      }
      if (!op->use_empty() || !mlir::isa<fir::ConvertOp, fir::EmboxOp, fir::AddrOfOp>(op)) continue;
      if (mlir::isa<fir::ConvertOp>(op))
        if (mlir::Operation* d = op->getOperand(0).getDefiningOp()) work.push_back(d);
      erased.insert(op);
      op->erase();
    }
  }

  /// Inline element stores: the same stores, into the local temp.
  static void rewriteInline(InlineCtor& c, unsigned id) {
    hlfir::DeclareOp local = makeLocalTemp(c.tmpDecl, c.tmpAlloc, c.seqTy, id);
    readFromTemp(c.exprs, local.getResult(0), c.seqTy.getEleTy());
    for (unsigned r = 0; r < c.tmpDecl->getNumResults(); ++r)
      for (mlir::OpOperand& use : llvm::make_early_inc_range(c.tmpDecl->getResult(r).getUses()))
        if (mlir::isa<hlfir::DesignateOp>(use.getOwner())) use.set(local->getResult(r));
    eraseHeapTemp(c.tmpDecl, c.tmpAlloc);
  }

  /// Erase a scratch slot nothing reads any more: every remaining user is a store into it.
  static void sweep(llvm::SetVector<mlir::Operation*>& slots) {
    for (mlir::Operation* slot : slots) {
      bool onlyStores = true;
      for (mlir::Operation* u : slot->getUsers())
        if (auto st = mlir::dyn_cast<fir::StoreOp>(u); !st || st.getMemref() != slot->getResult(0)) onlyStores = false;
      if (!onlyStores) continue;
      for (mlir::Operation* u : llvm::make_early_inc_range(slot->getUsers())) u->erase();
      slot->erase();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createMaterialiseArrayConstructorsPass() {
  return std::make_unique<MaterialiseArrayConstructorsPass>();
}

}  // namespace hlfir_bridge
