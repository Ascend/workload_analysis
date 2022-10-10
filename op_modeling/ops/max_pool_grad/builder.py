from xgboost import XGBRegressor
from ops.max_pool_grad.max_pool_grad_op import MaxPoolGradOp
from ops.max_pool_grad.max_pool_grad_op import MaxPoolGradIOGenerator
from ops.max_pool_grad.model import MaxPoolGradDetailFeature
from ops.max_pool_grad.model import MaxPoolGradOpModel
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf

@RegisterOf("MaxPoolGrad")
class MaxpoolV3Builder(XGBTrainingOpBuilder):
    dtypes = ['float16']
    formats = ['NHWC']
    io_generator = MaxPoolGradIOGenerator
    op = MaxPoolGradOp
    train_sample = 2000
    test_sample = 400
    model_pack = MaxPoolGradOpModel
    op_feature = MaxPoolGradDetailFeature
    xgb_estimator = XGBRegressor(
        learning_rate=0.1,
        n_estimators=400,
        max_depth=6,
        subsample=1,
        colsample_bytree=0.6,
    )