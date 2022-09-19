from xgboost import XGBRegressor
from ops.avg_pool_3d.avg_pool_3d_op import AvgPool3DOp
from ops.avg_pool_3d.avg_pool_3d_op import AvgPool3DIOGenerator
from ops.avg_pool_3d.model import AvgPool3DDetailFeature
from ops.avg_pool_3d.model import AvgPool3DOpModel
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf


@RegisterOf("AvgPool3DD")
class AvgPool3DBuilder(XGBTrainingOpBuilder):
    dtypes = ['float16']
    formats = ['NCDHW', 'NDHWC']
    io_generator = AvgPool3DIOGenerator
    model_pack = AvgPool3DOpModel
    op = AvgPool3DOp
    train_sample = 10000
    test_sample = 2000

    op_feature = AvgPool3DDetailFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.05,
        n_estimators=625,
        max_depth=6,
        subsample=0.6,
        colsample_bytree=1,
    )
