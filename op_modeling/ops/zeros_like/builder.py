from framework.op_register import RegisterOf
from template.builder.input_based_builder import SingleInOpBuilder
from ops.zeros_like.zeros_like_op import ZerosLikeOp
from template.model_packing.input_based_model import SingleInDtypeOpModel
from template.generator.random_shape_generator import RandomShapeSingleInOutGenerator
from template.feature.elementwise_feature import ElementwiseFlopsFeature

@RegisterOf("ZerosLike")
class ZerosLikeBuilder(SingleInOpBuilder):
    dtypes = ['float16', 'int8', 'int32', 'uint8']
    model_pack = SingleInDtypeOpModel
    io_generator = RandomShapeSingleInOutGenerator
    op_feature = ElementwiseFlopsFeature
    train_sample = 1000
    test_sample = 200
    op = ZerosLikeOp


