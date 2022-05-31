from termcolor import cprint

from workload.framework.init import init
from workload.framework.op_register import import_all_fusion_pass
from workload.framework.op_register import import_all_ops, OpManager, BuilderManager

if __name__ == '__main__':
    import_all_ops()
    init()
    op_model_builders = list()

    # keys = ["Add", 'MatMulV2']
    keys = ['MatMulV2']
    for key in keys:
        builder = BuilderManager.get(key)
        builder.data_collect()
        builder.modeling()
        builder.test()

    # 运行所有内容
    # for key_of_register in OpManager.keys():
    #     builder = BuilderManager.get(key_of_register)
    #     cprint(f"Building model of {key_of_register} by using {builder}", on_color='on_red')
    #     builder.data_collect()
    #     builder.modeling()
    #     builder.test()
