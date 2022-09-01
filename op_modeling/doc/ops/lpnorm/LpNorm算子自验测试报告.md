# 算子自验测试报告

## 概述
### 约束条件
+ 本报告由华为公司提供，用于描述相应特性的测试过程及结果。报告及内容版权归华为公司所有，非经授权，不可泄露给第三方。
+ 本报告部分章节为可选，请仔细阅读并填写正确的章节，对内容有疑问请及时联系华为众智接口人。
### 软件版本
+ CANN：5.0.4
### 测试目标
+ 验证LpNorm算子建模的训练过程及推理结果。
## 模型功能验证
### 执行入口脚本
`python  lpnorm/builder.py`

### 结果分析
#### 样本分布（数据采集正常必写，异常请忽略）
在此处说明数据采集的结果（以表格的形式呈现）
+ 说明采集的样本总量
+ 展现采集的数据类型，格式及形状的分布情况
+ 表格内容要能体现样本分布的广泛性

train_sample=10000, test_sample=2000

| 变量 | input | axes | p | keep_dims | epsilon |
|----|--------|--------|-----|-------|-------|
| 范围 | datatype: **float16, float** ,format: **ND** ,shape: **1D-5D**        | datatype: **list** |  0, 1, 2   |  datatype: **bool**     |  默认值1e-12     | 

#### 数据采集异常分析（数据采集异常必写，正常请忽略）
部分算子可能存在profiling数据无法采集的情况，开发者和华为众智接口人确认问题根源在于昇腾软硬件限制后，可以启动异常验收流程。本章节为此异常验收流程的主要交付内容之一。
##### 根因分析
+ 说明数据采集失败的原因（可以和华为众智接口人确认后填写）
##### 设计思路
阐述数据采集代码的设计思路
+ 数据生成的完整流程

#### 建模结果（数据采集正常且建模精度达标必写，其余请忽略）
+ RandomForestRegressor和XGBRegressor最好建模效果差别不大, 而经测试XGBRegressor速度快、针对大规模的数据效果更好, 模型容量更小,最终使用XGBRegressor建模
  
  + 使用的算子特征如下
  
    | 特征 | 说明|
    |----|--------|
    | x | np.prod(x_shape), 输入x_shape的乘积|
    | fc_shape1 | np.prod(x_shape) / np.prod(y_shape), axes指定维度的乘积|
    | y |np.prod(y_shape), 输出y_shape的乘积 | 
    | is_float16 |int(x['dtype'].lower() == "float16") |
    | is_float |int(x['dtype'].lower() == "float")) |
  
  + 模型参数如下
  
    | 参数 | 参数值|
    |----|--------|
    | learning_rate | 0.16|
    | n_estimators | 40|
    | max_depth | 20|
    | subsample | 0.7|
    | colsample_bytree | 0.9|
  
+ 模型的训练及测试精度

| 建模场景 | 训练集大小 | 80_percentage_error(取k折交叉验证的平均值) | 测试集大小 | 80_percentage_error | 模型超参数 |算子特征 |
|------|-------|----------------------------------|-------|---------------------|------|------|
|  p=0    |   6474    |       1.1501                           |   1304    |       1.1709              | learning_rate=0.16, n_estimators=40, max_depth=20, subsample=0.7, colsample_bytree=0.9   |    见上表  |
|  p=1    |    6712   |            1.5493                      |   1354    |          1.4028           |  learning_rate=0.16, n_estimators=40, max_depth=20, subsample=0.7, colsample_bytree=0.9    |    见上表  |
|  p=2    |   6586    |                 1.8044                 |   1340    |           1.8631        |    learning_rate=0.16, n_estimators=40, max_depth=20, subsample=0.7, colsample_bytree=0.9  |    见上表  |


#### 建模精度异常分析（数据采集正常且建模精度不达标必写，其余请忽略）
部分算子由于底层实现的复杂性，开发者难以建立达到目标精度的性能评估模型。开发者在与众智接口人协商确认后，可以启动异常验收流程。本章节为此异常验收流程的主要交付内容之一。
+ 说明建模过程中尝试过的各个模型（包括超参数）及使用的算子特征
+ 说明各模型的训练及测试精度

建模过程中尝试了RandomForestRegressor和XGBRegressor进行建模, 对p=0, 1, 2分场景建模的情况下, RandomForestRegressor效果和XGBRegressor差异不大

  两个模型使用的算子特征相同，如下：

  | 特征 | 说明|
  |----|--------|
  | x | np.prod(x_shape), 输入x_shape的乘积|
  | fc_shape1 | np.prod(x_shape) / np.prod(y_shape), axes指定维度的乘积|
  | y |np.prod(y_shape), 输出y_shape的乘积 |
  | is_float16 |int(x['dtype'].lower() == "float16") |
  | is_float |int(x['dtype'].lower() == "float")) |
    
+ XGBRegressor最好效果

  | 建模场景 | 训练集大小 | 80_percentage_error(取k折交叉验证的平均值) | 测试集大小 | 80_percentage_error | 模型超参数 |算子特征 |
  |------|-------|----------------------------------|-------|---------------------|------|------|
  |  p=0    |   6474    |       1.1501                           |   1304    |       1.1709              | learning_rate=0.16, n_estimators=40, max_depth=20, subsample=0.7, colsample_bytree=0.9   |    见上表  |
  |  p=1    |    6712   |            1.5493                      |   1354    |          1.4028           |   learning_rate=0.16, n_estimators=40, max_depth=20, subsample=0.7, colsample_bytree=0.9    |    见上表  |
  |  p=2    |   6586    |                 1.8044                 |   1340    |           1.8631        |    learning_rate=0.16, n_estimators=40, max_depth=20, subsample=0.7, colsample_bytree=0.9   |    见上表  |

+ RandomForestRegressor最好效果

  | 建模场景 | 训练集大小 | 80_percentage_error(取k折交叉验证的平均值) | 测试集大小 | 80_percentage_error | 模型超参数 |算子特征 |
  |------|-------|----------------------------------|-------|---------------------|------|------|
  |  p=0    |   6474    |           1.1561                       |    1304   |            1.1574         |   n_estimators=30, max_depth=45   |    见上表  |
  |  p=1    |   6712    |              1.5332                   |   1354    |            1.4232        |    n_estimators=30, max_depth=45  |   见上表   |
  |  p=2    |   6586    |               1.9243                   |    1340   |            1.8206         |     n_estimators=30, max_depth=45 |   见上表   |

  建模过程中尝试过的各个模型的超参数、详细模型调优过程请见LpNorm算子模型而分析报.md

### 测试结论
针对建模结果，在以下三条结论中选择对应的那一条
+ 算子性能评估模型训练成功，但建模精度不达标
