from xgboost import XGBRegressor
from ops.max_pool_v3.max_pool_v3_op import MaxPoolV3Op
from ops.max_pool_v3.max_pool_v3_op import MaxPoolV3IOGenerator
from ops.max_pool_v3.model import MaxPoolV3DetailFeature
from ops.max_pool_v3.model import MaxPoolV3OpModel
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf

@RegisterOf("MaxPoolV3")
class MaxpoolV3Builder(XGBTrainingOpBuilder):
    dtypes = ['float16']
    formats = ['NHWC']
    io_generator = MaxPoolV3IOGenerator
    op = MaxPoolV3Op
    train_sample = 2000
    test_sample = 400
    model_pack = MaxPoolV3OpModel
    op_feature = MaxPoolV3DetailFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.1,
        n_estimators=400,
        max_depth=6,
        subsample=1,
        colsample_bytree=0.6,
    )
