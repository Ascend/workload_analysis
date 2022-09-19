# 算子自验测试报告

## 概述
### 约束条件
+ 本报告由华为公司提供，用于描述相应特性的测试过程及结果。报告及内容版权归华为公司所有，非经授权，不可泄露给第三方。
+ 本报告部分章节为可选，请仔细阅读并填写正确的章节，对内容有疑问请及时联系华为众智接口人。
### 软件版本
+ CANN：5.0.4
### 测试目标
+ 验证InstanceNorm算子建模的训练过程及推理结果。
## 模型功能验证
### 执行入口脚本
`python  instancenorm/builder.py`

### 结果分析
#### 样本分布（数据采集正常必写，异常请忽略）
在此处说明数据采集的结果（以表格的形式呈现）
+ 说明采集的样本总量
+ 展现采集的数据类型，格式及形状的分布情况
+ 表格内容要能体现样本分布的广泛性

train_sample=4500, test_sample=1500

| 变量 | x | gamma | beta | data_format | epsilon |
|----|--------|--------|-----|-------|-------|
| 范围 | datatype: **float16, float** ,format: **ND** ,shape: **>=3D**        | datatype: **float16, float** | datatype: **float16, float** |  datatype: **String**  |  默认值1e-12     |

#### 数据采集异常分析（数据采集异常必写，正常请忽略）
部分算子可能存在profiling数据无法采集的情况，开发者和华为众智接口人确认问题根源在于昇腾软硬件限制后，可以启动异常验收流程。本章节为此异常验收流程的主要交付内容之一。
##### 根因分析
+ 说明数据采集失败的原因   
由于底层实现的复杂性，mean与variance存在差异，采集数据过程中偶现报错，数据采集越多，报错越频繁。   
和华为众智接口人对齐后确认该算子的数据采集采用特殊方法，以当前4500条数据作为训练测试集
##### 设计思路
阐述数据采集代码的设计思路
+ 创建`InstanceNormOp`实例，继承`OpBase`，并设置输入输出参数。   
+ 创建`InstanceNormIOGenerator`实例，基于`IOGenerator`实现生成类`RandomShapeValueGenerator`,随机种子固定为常数123。   
+ 其中输入参数x的shape由方法`get_shape_strategy`直接生成，输入参数gamma、beta和输出y以及输出参数mean、variance则根据x生成。对于epsilon，它的shape恒为[1]，value为1e-6。   
+ 通过`get_io_strategys`对input、output以及attr的shape和value进行整合，返回其tensor。   

#### 建模结果（数据采集正常且建模精度达标必写，其余请忽略）
+ RandomForestRegressor和XGBRegressor最好建模效果差别不大, 而经测试XGBRegressor速度快、针对大规模的数据效果更好, 模型容量更小,最终使用XGBRegressor建模
  
  + 使用的算子特征如下
  
  + 
    | 特征 | 说明|
    |----|--------|
    | x | np.prod(x_shape), 输入x_shape的乘积|
    | x_mean | np.prod(mean_shape), 输出mean_shape的乘积         |
    | y |np.prod(y_shape), 输出y_shape的乘积 |
    | means |np.prod(mean_shape), 输出mean_shape的乘积 |
    | variance |np.prod(variance_shape), 输出variance_shape的乘积 |
    | variance_e |np.prod(variance_shape), 输出variance_shape的乘积 |
    | de |np.prod(x_shape), 输入x_shape的乘积 |
    | mul |np.prod(x_shape), 输入x_shape的乘积 |
    | add |np.prod(x_shape), 输入x_shape的乘积 |
    | is_float16 |int(x['dtype'].lower() == "float16") |
    | is_float |int(x['dtype'].lower() == "float")) |
  
  + 模型参数如下
  
    | 参数 | 参数值|
    |----|--------|
    | learning_rate | 0.15 |
    | n_estimators | 320 |
    | max_depth | 3 |
    | subsample | 0.7|
    | colsample_bytree | 0.8 |
  
+ 模型的训练及测试精度

| 建模算子     | 训练集大小 | 80_percentage_error(取k折交叉验证的平均值) | 测试集大小 | 80_percentage_error | 模型超参数                                                   |
| ------------ | ---------- | ------------------------------------------ | ---------- | ------------------- | ------------------------------------------------------------ |
| InstanceNorm | 4500       | 2.04                                       | 1500       | 1.96                | learning_rate=0.15, n_estimators=320, max_depth=3, subsample=0.7, colsample_bytree=0.8 |


#### 建模精度异常分析
部分算子由于底层实现的复杂性，开发者难以建立达到目标精度的性能评估模型。开发者在与众智接口人协商确认后，可以启动异常验收流程。本章节为此异常验收流程的主要交付内容之一。
+ 说明建模过程中尝试过的各个模型（包括超参数）及使用的算子特征
+ 说明各模型的训练及测试精度

建模过程中尝试了RandomForestRegressor和XGBRegressor进行建模,, RandomForestRegressor效果和XGBRegressor差异不大

  两个模型使用的算子特征相同，如下：


| 特征 | 说明|重要性 |
|----|--------|--------|
| x | np.prod(x_shape), 输入x_shape的乘积|0.073494  |
| x_mean | np.prod(mean_shape), 输出mean_shape的乘积         | 0.071685 |
| y |np.prod(y_shape), 输出y_shape的乘积 | 0.069854 |
| variance_e |np.prod(variance_shape), 输出variance_shape的乘积 | 0.538009 |
| means |np.prod(mean_shape), 输出mean_shape的乘积 |
| variance |np.prod(variance_shape), 输出variance_shape的乘积 |
| de |np.prod(x_shape), 输入x_shape的乘积 | 0.071885 |
| mul |np.prod(x_shape), 输入x_shape的乘积 | 0.073312 |
| add |np.prod(x_shape), 输入x_shape的乘积 | 0.073236 |
| is_float16 |int(x['dtype'].lower() == "float16") | 0.014796 |
| is_float |int(x['dtype'].lower() == "float")) | 0.013730 |

+ XGBRegressor最好效果

| 建模算子     | 训练集大小 | 80_percentage_error(取k折交叉验证的平均值) | 测试集大小 | 80_percentage_error | 模型超参数                                                   |
| ------------ | ---------- | ------------------------------------------ | ---------- | ------------------- | ------------------------------------------------------------ |
| InstanceNorm | 4500       | 2.04                                       | 1500       | 1.96                | learning_rate=0.15, n_estimators=320, max_depth=3, subsample=0.7, colsample_bytree=0.8 |

- RandomForestRegressor最好效果

| 建模场景 | 训练集大小 | 80_percentage_error(取k折交叉验证的平均值) | 测试集大小 | 80_percentage_error | 模型超参数                     |
| -------- | ---------- | ------------------------------------------ | ---------- | ------------------- | ------------------------------ |
| p=0      | 4500       | 1.99                                       | 1500       | 1.98                | n_estimators=200, max_depth=45 |

建模过程中尝试过的各个模型的超参数、详细模型调优过程请见InstanceNorm算子模型而分析报.md

### 测试结论
针对建模结果，在以下三条结论中选择对应的那一条
+ 算子性能评估模型训练成功，但建模精度不达标
