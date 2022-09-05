#!/usr/bin/python
# -*- coding: UTF-8 -*-
from framework.op_register import RegisterOf
from ops.l2_loss.l2_loss_op import L2LossOp
from ops.l2_loss.model import L2LossDetailFeature
from ops.l2_loss.l2_loss_op import L2LossIOGenerator
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from xgboost import XGBRegressor
from template.model_packing.input_based_model import SingleInDtypeOpModel


@RegisterOf("L2Loss")
class L2LossBuilder(XGBTrainingOpBuilder):
    dtypes = ['float', 'float16']
    train_sample = 1000
    test_sample = 200
    model_pack = SingleInDtypeOpModel
    op = L2LossOp
    io_generator = L2LossIOGenerator
    op_feature = L2LossDetailFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.02,
        n_estimators=500,
        max_depth=5,
        subsample=1,
        colsample_bytree=0.8
    )