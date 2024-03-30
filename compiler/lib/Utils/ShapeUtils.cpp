//===- ShapeUtils.cpp ------------------------------------*--- C++ -*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/ShapeUtils.h"
#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Dialect/Shape/Transforms/InsertTieShape.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Transforms/ShapeReification.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <queue>
#include <string>

using namespace mlir;
using namespace llvm;

namespace mlir {

namespace {

SmallVector<Operation *> collectAllShapeOpsForReturn(Operation *retOp) {
  llvm::DenseSet<Operation *> visitedOp;
  std::queue<Operation *> opQueue;

  opQueue.push(retOp);
  while (!opQueue.empty()) {
    auto frontOp = opQueue.front();
    opQueue.pop();
    if (visitedOp.find(frontOp) != visitedOp.end()) {
      continue;
    }
    visitedOp.insert(frontOp);
    for (Value operand : frontOp->getOperands()) {
      if (!operand.getDefiningOp()) {
        continue;
      }
      if (Operation *defOp = operand.getDefiningOp()) {
        opQueue.push(defOp);
      }
    }
  }
  visitedOp.erase(retOp);
  return SmallVector<Operation *>(visitedOp.begin(), visitedOp.end());
}

bool deducedFromFuncArgShape(Value value) {
  if (value.isa<BlockArgument>()) {
    return false;
  }

  auto defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }

  if (isa<arith::ConstantIndexOp, arith::ConstantOp>(defOp)) {
    return true;
  }

  if (isa<tensor::DimOp, shape::ShapeOfOp>(defOp)) {
    auto operand = defOp->getOperand(0);
    if (operand.isa<BlockArgument>()) {
      return true;
    }
    return false;
  }

  for (Value &&operand : defOp->getOperands()) {
    if (!deducedFromFuncArgShape(operand)) {
      return false;
    }
  }
  return true;
}

LogicalResult reifyCallOp(OpBuilder &builder, Operation *op,
                          SmallVectorImpl<Value> &reifications) {
  OpBuilder::InsertionGuard guard(builder);
  auto callOp = dyn_cast<func::CallOp>(op);
  if (!callOp) {
    return failure();
  }

  ModuleOp moduleOp = op->getParentRegion()->getParentOfType<ModuleOp>();
  // auxiliary builder used for create operations in shape func
  // original builder maybe a rewriter, used for create operations in specific pattern.
  OpBuilder auxiliaryBuilder(moduleOp);
  StringRef funcName = callOp.getCallee();
  auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);

  // clone funcOp, newFuncOp used for deduce function shape
  std::string newFuncName = funcName.str() + "_Shape";
  auxiliaryBuilder.setInsertionPointToStart(moduleOp.getBody());
  auto newFuncOp = auxiliaryBuilder.create<func::FuncOp>(
      funcOp->getLoc(), newFuncName, funcOp.getFunctionType());
  newFuncOp.setPrivate();
  IRMapping emptyBvm;
  funcOp.cloneInto(newFuncOp, emptyBvm);

  // replace the operands of returnOp with corresponding shape
  func::ReturnOp retOp = *newFuncOp.getOps<func::ReturnOp>().begin();
  if (!retOp) {
    newFuncOp->erase();
    return failure();
  }

  SmallVector<Type> allResultTypes;
  SmallVector<Value> allResults;

  auxiliaryBuilder.setInsertionPoint(retOp);
  for (Value &&retTensor : retOp.getOperands()) {
    auto retShape =
        auxiliaryBuilder.create<shape::ShapeOfOp>(retOp.getLoc(), retTensor);
    allResultTypes.emplace_back(retShape.getType());
    allResults.emplace_back(retShape);
  }

  // return the shape of original tensor returned by function
  auto newRetOp =
      auxiliaryBuilder.create<func::ReturnOp>(retOp.getLoc(), allResults);
  auto newFuncType = auxiliaryBuilder.getFunctionType(
      newFuncOp.getArgumentTypes(), allResultTypes);
  newFuncOp.setFunctionType(newFuncType);
  retOp->erase();

  // reify newFunc to get the shape computation for current callOp
  {
    PassManager pm(moduleOp->getContext(), func::FuncOp::getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createByteIRShapeReificationPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (mlir::failed(pm.run(newFuncOp))) {
      newFuncOp->erase();
      return failure();
    }
  }

  // collect all shape computation ops
  SmallVector<Operation *> reificationOps =
      collectAllShapeOpsForReturn(newRetOp);

  // value can only depends on the shape of FuncArgs.
  for (Value &&ret : newRetOp.getOperands()) {
    if (!deducedFromFuncArgShape(ret)) {
      newFuncOp->erase();
      return failure();
    }
  }

  // mapping the shape computation ops and collect reifications
  {
    mlir::computeTopologicalSorting(reificationOps);

    IRMapping bvm;
    size_t numArg = newFuncOp.getNumArguments();
    for (size_t i = 0; i < numArg; ++i) {
      bvm.map(newFuncOp.getArgument(i), callOp.getOperand(i));
    }

    builder.setInsertionPoint(callOp);

    for (Operation *oldOp : reificationOps) {
      auto newOp = builder.clone(*oldOp, bvm);
    }

    for (Value &&ret : newRetOp.getOperands()) {
      reifications.push_back(bvm.lookup(ret));
    }
  }

  // remove newFuncOp
  newFuncOp->erase();
  return success();
}

} // namespace

LogicalResult reifyShapes(OpBuilder &builder, Operation *op,
                          SmallVectorImpl<Value> &reifications) {
  if (!op)
    return failure();

  if (op->hasTrait<hlo::OpTrait::CompatibleOperandsAndResultType>()) {
    // CompatibleOperandsAndResultType does not implement reify
    reifications.push_back(
        builder.create<shape::ShapeOfOp>(op->getLoc(), op->getOperand(0)));
    return success();
  }

  if (auto origin = dyn_cast<InferShapedTypeOpInterface>(op)) {
    if (failed(origin.reifyReturnTypeShapes(builder, origin->getOperands(),
                                            reifications))) {
      return failure();
    }
  } else if (auto reifyFunc =
                 reifyReturnTypeShapes(op->getName().getStringRef())) {
    if (failed(reifyFunc(op, builder, op->getOperands(), reifications))) {
      return failure();
    }
  } else if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
    auto inferFunc = reifyReturnTypeShapes(customCall.getCallTargetName());
    if (!inferFunc) {
      return failure();
    }
    if (failed(inferFunc(op, builder, op->getOperands(), reifications)))
      return failure();
  } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
    if (failed(reifyCallOp(builder, op, reifications))) {
      return failure();
    }
  } else if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    for (OpResult &&result : op->getOpResults()) {
      auto tiedOperand = dpsOp.getTiedOpOperand(result);
      reifications.push_back(
          builder.create<shape::ShapeOfOp>(op->getLoc(), tiedOperand->get()));
    }
  } else {
    // Return failure if op doesn't have InferShapedTypeOpInterface and not
    // registered.
    return failure();
  }

  return success();
}

FailureOr<SmallVector<Value>> createEmptyTensorForResult(OpBuilder &builder,
                                                         Operation *op) {
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Value> emptyTensors;
  bool resultsHasDynamicShape = false;
  for (auto &&result : op->getResults()) {
    if (auto resType = result.getType().template dyn_cast<ShapedType>()) {
      if (resType.hasStaticShape()) {
        auto emptyOp = builder.create<tensor::EmptyOp>(
            op->getLoc(), resType.getShape(), resType.getElementType());
        emptyTensors.emplace_back(emptyOp);
      } else {
        resultsHasDynamicShape = true;
        break;
      }
    }
  }

  if (resultsHasDynamicShape) {
    emptyTensors.clear();
    registerAllMhloReifyReturnTypeShapes();
    SmallVector<Value, 1> reifications;

    if (mlir::reifyShapes(builder, op, reifications).failed()) {
      return failure();
    }

    for (auto &&resultAndShape : llvm::zip(op->getResults(), reifications)) {
      SmallVector<Value, 1> dynamicSizes;
      auto resType = std::get<0>(resultAndShape).getType().cast<ShapedType>();
      for (size_t i = 0; i < resType.getRank(); ++i) {
        if (resType.isDynamicDim(i)) {
          auto dim = builder
                         .create<tensor::ExtractOp>(
                             op->getLoc(), std::get<1>(resultAndShape),
                             ValueRange{builder.create<arith::ConstantIndexOp>(
                                 op->getLoc(), static_cast<int64_t>(i))})
                         .getResult();
          dynamicSizes.emplace_back(dim);
        }
      }
      auto emptyOp =
          builder.create<tensor::EmptyOp>(op->getLoc(), resType, dynamicSizes);
      emptyTensors.emplace_back(emptyOp);
    }
  }
  return emptyTensors;
}

} // namespace mlir
