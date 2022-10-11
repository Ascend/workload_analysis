from xgboost import XGBRegressor
from ops.max_pool_3d.max_pool_3d_op import MaxPool3DOp
from ops.max_pool_3d.max_pool_3d_op import MaxPool3DIOGenerator
from ops.max_pool_3d.model import MaxPool3DDetailFeature
from ops.max_pool_3d.model import MaxPool3DOpModel
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf


@RegisterOf("MaxPool3D")
class MaxPool3DBuilder(XGBTrainingOpBuilder):
    dtypes = ['float16']
    formats = ['NDHWC']
    io_generator = MaxPool3DIOGenerator
    model_pack = MaxPool3DOpModel
    op = MaxPool3DOp
    train_sample = 2000
    test_sample = 400

    op_feature = MaxPool3DDetailFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.05,
        n_estimators=350,
        max_depth=6,
        subsample=0.6,
        colsample_bytree=1,
    )
