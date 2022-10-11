from xgboost import XGBRegressor
from ops.max_pool.max_pool_op import MaxPoolOp
from ops.max_pool.max_pool_op import MaxPoolIOGenerator
from ops.max_pool.model import MaxPoolDetailFeature
from ops.max_pool.model import MaxPoolOpModel
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf

@RegisterOf("MaxPool")
class MaxPoolBuilder(XGBTrainingOpBuilder):
    dtypes = ['float16']
    formats = ['NHWC']
    io_generator = MaxPoolIOGenerator
    op = MaxPoolOp
    train_sample = 2000
    test_sample = 400
    model_pack = MaxPoolOpModel
    op_feature = MaxPoolDetailFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.11,
        n_estimators=400,
        max_depth=4,
        subsample=1,
        colsample_bytree=0.9,
    )