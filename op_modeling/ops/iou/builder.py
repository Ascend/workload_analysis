from xgboost import XGBRegressor
from framework.op_register import RegisterOf
from ops.iou.iou_op import IouIOGenerator, IouOp
from ops.iou.model import IouDetailFeature, IouOpModel
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
@RegisterOf("Iou")
class IouBuilder(XGBTrainingOpBuilder):
    dtypes = ["float16"] 
    formats = ["ND"]
    io_generator = IouIOGenerator
    model_pack = IouOpModel
    op = IouOp
    train_sample = 10000
    test_sample = 2000
    op_feature = IouDetailFeature

    xgb_estimator = XGBRegressor(
        learning_rate=0.11,
        n_estimators=500,
        max_depth=6,
        subsample=1,
        colsample_bytree=0.9,
    )

