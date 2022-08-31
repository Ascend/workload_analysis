# ExpandDims算子自验测试报告

## 概述
### 约束条件
+ 本报告由华为公司提供，用于描述相应特性的测试过程及结果。报告及内容版权归华为公司所有，非经授权，不可泄露给第三方。
+ 本报告部分章节为可选，请仔细阅读并填写正确的章节，对内容有疑问请及时联系华为众智接口人。
### 软件版本
+ CANN：5.0.4
### 测试目标
+ 验证ExpandDims算子建模的训练过程及推理结果。
## 模型功能验证
### 执行入口脚本
`python  expand_dims/builder.py`

### 结果分析

#### 数据采集异常分析
##### 根因分析
+ 与华为众智接口人确认为GE算子，无法采集profiling 数据。
##### 设计思路
+ 创建`ExpandDimsOp`实例，继承`OpBase`，并设置输入输出参数。
+ 创建`ExpandDimsIOGenerator`实例，基于`IOGenerator`实现生成类`RandomShapeValueGenerator`
+ 其中输入参数x的shape由方法`get_shape_strategy`直接生成，输入参数axis和输出参数y则根据x生成。对于axis，它的shape恒为[1]，value的生成规则为0到x的dim之间的随机值。对于y，则根据算子规则和输入参数生成其shape。
+ 通过`get_io_strategys`对axis的shape和value进行整合，返回其tensor。

### 测试结论
+ 算子性能数据采集失败，无法进行算子建模