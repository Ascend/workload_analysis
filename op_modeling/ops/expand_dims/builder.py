from xgboost import XGBRegressor
from ops.expand_dims.expand_dims_op import ExpandDimsOp
from ops.expand_dims.expand_dims_op import ExpandDimsIOGenerator
from ops.expand_dims.model import ExpandDimsDetailFeature
from ops.expand_dims.model import ExpandDimsOpModel
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf


@RegisterOf("ExpandDims")
class ExpandDimsBuilder(XGBTrainingOpBuilder):
    dtypes = ['int32']
    formats = ['ND']
    io_generator = ExpandDimsIOGenerator
    model_pack = ExpandDimsOpModel
    op = ExpandDimsOp
    train_sample = 100
    test_sample = 100

    op_feature = ExpandDimsDetailFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.11,
        n_estimators=500,
        max_depth=6,
        subsample=1,
        colsample_bytree=0.9,
    )
