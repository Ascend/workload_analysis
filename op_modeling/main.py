from termcolor import cprint

from framework.init import init
from framework.op_register import import_all_ops, BuilderManager

if __name__ == '__main__':
    import_all_ops()
    init()
    op_model_builders = list()

    keys = ['Add', 'FullyConnection']
    # keys = ['Add']
    for key in keys:
        builder = BuilderManager.get(key)
        builder.data_collect()
        builder.modeling()
        builder.pack()
        builder.test()

    # 运行所有内容
    # for key_of_register in OpManager.keys():
    #     builder = BuilderManager.get(key_of_register)
    #     cprint(f"Building model of {key_of_register} by using {builder}", on_color='on_red')
    #     builder.data_collect()
    #     builder.modeling()
    #     builder.test()
