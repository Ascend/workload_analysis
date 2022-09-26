# MaxPoolGrad算子自验测试报告

## 概述
### 约束条件
+ 本报告由华为公司提供，用于描述相应特性的测试过程及结果。报告及内容版权归华为公司所有，非经授权，不可泄露给第三方。
+ 本报告部分章节为可选，请仔细阅读并填写正确的章节，对内容有疑问请及时联系华为众智接口人。
### 软件版本
+ CANN：5.0.4
### 测试目标
+ 验证MaxPoolGrad算子建模的训练过程及推理结果。
## 模型功能验证
### 执行入口脚本
`python  max_pool_grad/builder.py`

### 结果分析

#### 数据采集异常分析

##### 根因分析

+ 与华为众智接口人确认,在aic-ascend310-ops-info.json中查到该算子在310芯片上是AICPU算子，无法采集profiling 数据。

##### 设计思路

+ 创建MaxPoolGradOp实例，继承OpBase，并设置输入输出参数。
+ 创建MaxPoolGradIOGenerator实例，基于IOGenerator实现生成类RandomShapeValueGenerator。
+ 其中输入参数x1的shape由方法get_shape_strategy直接生成，padding在SAME与VALID间随机生成。输入参数x2与grad，属性ksize与strides，输出参数y均根据x1以及padding在gen_strategies函数中生成。首先生成ksize，ksize= [1, ksize_h_value, ksize_w_value, 1]，当padding为SAME时，其中ksize_h_value以及ksize_w_value取值范围为3~10，当padding为VALID时，其中ksize_h_value（ksize_w_value）取值范围为3~x_shape[1]（x_shape[2]）。其次生成stride，stride = [1, strides_h_value, strides_w_value, 1]，其中strides_h_value（strides_w_value）取值范围为1~ksize_h_value（ksize_w_value）。x2的shape为[x1_shape[0], x2_h_value, x2_w_value, x1_shape[3]]，其中x2_h_value = math.ceil(x1_shape[1] / strides_h_value)，x2_w_value = math.ceil(x1_shape[2] / strides_w_value)。grad与x2的shape相同。输出参数y与x1同shape且同类型，y_strategy = x1_strategy。
+ data_format默认为NHWC，通过get_io_strategys进行整合，返回其tensor。

### 测试结论
+ 算子性能评估模型训练成功，且建模精度达标。