from xgboost import XGBRegressor
from ops.topk.topk_op import TopKOp
from ops.topk.topk_op import TopKIOGenerator
from ops.topk.model import TopKDetailFeature
from ops.topk.model import TopKOpModel
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf


@RegisterOf("TopKD")
class TopKBuilder(XGBTrainingOpBuilder):
    dtypes = ['float16']
    formats = ['ND']
    io_generator = TopKIOGenerator
    model_pack = TopKOpModel
    op = TopKOp
    train_sample = 2000
    test_sample = 400
    op_feature = TopKDetailFeature

    xgb_estimator = XGBRegressor(
        learning_rate=0.10,
        n_estimators=500,
        max_depth=4,
        subsample=1,
        colsample_bytree=0.9,
    )