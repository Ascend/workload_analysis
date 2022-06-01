def linear_curve(x, w, b):
    """
    线性函数
    """
    return w * x + b


def data_type_one_hot(x, w01, w1, w02, w2):
    """
    组合特征构建方式1
    """
    return (w01 * x[:, 0] + w1) * x[:, 1] + (w02 * x[:, 0] + w2) * x[:, 2]


def data_type_combine(x, w0, w1, b):
    """
    组合特征构建方式2
    """
    return w0 * x[:, 0] + w1 * x[:, 1] + b


def quadratic_curve(x, a, b, c):
    """
    二次函数
    """
    return a * (x ** 2) + b * x + c
