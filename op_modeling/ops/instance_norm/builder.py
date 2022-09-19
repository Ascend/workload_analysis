from ops.instance_norm.instance_norm_op import InstanceNormOp
from ops.instance_norm.instance_norm_op import InstanceNormIOGenerator
from ops.instance_norm.model import InstanceNormFeature
from ops.instance_norm.model import InstanceNormOpModel
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


@RegisterOf("InstanceNorm")
class InstanceNormBuilder(XGBTrainingOpBuilder):
    dtypes = ['float16', 'float']
    formats = ["ND"]
    train_sample = 19000
    test_sample = 2000
    io_generator = InstanceNormIOGenerator
    model_pack = InstanceNormOpModel
    op = InstanceNormOp
    op_feature = InstanceNormFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.15,
        n_estimators=320,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.8,
    )

