from template.generator.vector.single_op_generator import SingleVectorIOGenerator


class BinaryVectorIOGenerator(SingleVectorIOGenerator):
    """
    该类主要返回算子执行需要的输入输出内容，并支持被算子定义所使用
    该类需要融合Tensor策略以及组合的工作
    """

    def __init__(self, dtype, is_training=True, n_sample=200):
        super(BinaryVectorIOGenerator, self).__init__(dtype, is_training, n_sample)

    def get_io_strategys(self, *assemble_strategy):
        # 此处表示针对add类算子而言，x,y,z的生成策略一致
        x, y = super().get_io_strategys(*assemble_strategy)
        return [x, x, y]


