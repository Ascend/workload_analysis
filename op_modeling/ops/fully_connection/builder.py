from xgboost import XGBRegressor
from ops.fully_connection.fully_connection_op import FullyConnectionOp
from ops.fully_connection.fully_connection_op import FullyConnectionIOGenerator
from ops.fully_connection.model import FullyConnectionDetailFeature
from ops.fully_connection.model import FullyConnectionOpModel
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf


@RegisterOf("FullyConnection")
class FullyConnectionBuilder(XGBTrainingOpBuilder):
    dtypes = ['float16', 'int8']
    formats = ['NCHW']
    io_generator = FullyConnectionIOGenerator
    model_pack = FullyConnectionOpModel
    op = FullyConnectionOp
    train_sample = 10000
    test_sample = 2000

    op_feature = FullyConnectionDetailFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.11,
        n_estimators=500,
        max_depth=6,
        subsample=1,
        colsample_bytree=0.9,
    )
