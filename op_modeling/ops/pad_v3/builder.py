from xgboost import XGBRegressor
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf
from ops.pad_v3.pad_v3_op import PadV3IOGenerator, PadV3Op
from ops.pad_v3.model import PadV3OpModel
from ops.pad_v3.model import PadV3DetailFeature

@RegisterOf("PadV3")
class PadV3Builder(XGBTrainingOpBuilder):
    dtypes=['float16', 'float']
    formats = ['NCHW']
    io_generator = PadV3IOGenerator
    model_pack = PadV3OpModel
    op = PadV3Op
    train_sample = 1200
    test_sample = 240
    op_feature = PadV3DetailFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=6,
        subsample=1,
        colsample_bytree=0.9,
    )