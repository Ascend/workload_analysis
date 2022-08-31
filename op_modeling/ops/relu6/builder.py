from framework.op_register import RegisterOf
from ops.relu6.relu6_op import Relu6Op
from template.builder.input_based_builder import SingleInOpBuilder
from template.model_packing.input_based_model import SingleInDtypeOpModel


@RegisterOf("Relu6")
class Relu6Builder(SingleInOpBuilder):
    dtypes = ['float16', 'float', 'int32']
    model_pack = SingleInDtypeOpModel
    train_sample = 1000
    test_sample = 200
    op = Relu6Op
