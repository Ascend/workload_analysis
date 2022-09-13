from xgboost import XGBRegressor
from ops.maxpoolgrad.maxpoolgrad_op import MaxPoolGradOp
from ops.maxpoolgrad.maxpoolgrad_op import MaxPoolGradIOGenerator
# from ops.MaxPool.model import MaxPoolDetailFeature
# from ops.MaxPool.model import MaxPoolOpModel
from template.builder.xgb_training_op_builder import XGBTrainingOpBuilder
from framework.op_register import RegisterOf

@RegisterOf("MaxPoolGrad")
class MaxPoolBuilder(XGBTrainingOpBuilder):
    dtypes = ['float16']
    formats = ['NHWC']
    io_generator = MaxPoolGradIOGenerator
    op = MaxPoolGradOp
    # train_sample = 3000
    # test_sample = 600
    #
    # model_pack = MaxPoolOpModel
    # op_feature = MaxPoolDetailFeature
    # xgb_estimator = XGBRegressor(
    #     learning_rate=0.11,
    #     n_estimators=200,
    #     max_depth=6,
    #     subsample=1,
    #     colsample_bytree=0.9,
    # )