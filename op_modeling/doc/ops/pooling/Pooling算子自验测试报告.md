# 算子自验测试报告

## 概述
### 约束条件
+ 本报告由华为公司提供，用于描述相应特性的测试过程及结果。报告及内容版权归华为公司所有，非经授权，不可泄露给第三方。
+ 本报告部分章节为可选，请仔细阅读并填写正确的章节，对内容有疑问请及时联系华为众智接口人。
### 软件版本
+ CANN：5.0.4
### 测试目标
+ 验证Pooling算子建模的训练过程及推理结果。
## 模型功能验证
### 执行入口脚本
`python  pooling/builder.py`

### 结果分析
#### 样本分布（数据采集正常必写，异常请忽略）
在此处说明数据采集的结果（以表格的形式呈现）
+ 说明采集的样本总量
+ 展现采集的数据类型，格式及形状的分布情况
+ 表格内容要能体现样本分布的广泛性


| 变量 | x    | mode   | global_pooling  | window  | stride  | pad |  dilation   |  ceil_mode   |  data_format   | y  |
|----|--------------------------------------------------------|----------------------------------------------------|--------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------------------------|-----|-----|-----|-----|----|
| 范围 | datatype: **float16** ,format: **NCHW** ,shape: **4D** | datatype: **int32** ,format: **ND** ,shape: **1D** | datatype: **bool** ,format: **ND** ,shape: **1D**  Defaults to "false". | datatype: **listInt** ,format: **ND** ,shape: **2D** Defaults to [1, 1] | datatype: **listInt** ,format: **ND** ,shape: **2D**  Defaults to [1, 1] |   datatype: **listInt** ,format: **ND** ,shape: **4D**  Defaults to [0, 0, 0, 0]    |  datatype: **listInt** ,format: **ND** ,shape: **4D**  Defaults to [1, 1, 1, 1]   |   datatype: **int32** ,format: **ND** ,shape: **1D**  |  默认值仅为："NCHW"   |   datatype: **float16** ,format: **NCHW** ,shape: **4D**  |

#### 建模结果（数据采集正常且建模精度达标必写，其余请忽略）
+ 说明最终使用的模型（包括超参数）及使用的算子特征
+ 说明模型的训练及测试精度
该章节内容以表格的形式呈现，参考的表格如下，其中不同建模场景指对算子样本空间的划分（如示例Add算子按datatype进行建模场景区分）

| 建模场景                    | 训练集大小 | 80_percentage_error(取k折交叉验证的平均值) | 测试集大小 | 80_percentage_error | 模型超参数 | 算子特征                                                                                       |
|-------------------------|-------|----------------------------------|-------|---------------------|------|--------------------------------------------------------------------------------------------|
| mode=1，global_pooling=1 | 4000  | 0.042286                         | 2000  | 0.043767            |  learning_rate=0.11,n_estimators=350,max_depth=6,subsample=0.9,colsample_bytree=1    | flops,N,Cin,x_h,x_w,y_h,y_w,window_h,window_w,stride_h,stride_w,pad_top,pad_left,ceil_mode |
| mode=0，global_pooling=1 | 4000  | 0.0817396                        | 2000  | 0.077419            |  learning_rate=0.11,n_estimators=350,max_depth=6,subsample=0.9,colsample_bytree=1    | flops,N,Cin,x_h,x_w,y_h,y_w,window_h,window_w,stride_h,stride_w,pad_top,pad_left,ceil_mode  |
| mode=0，global_pooling=0 | 4000  | 0.1753534                        | 2000  | 0.166518            |  learning_rate=0.11,n_estimators=350,max_depth=6,subsample=0.9,colsample_bytree=1    | flops,N,Cin,x_h,x_w,y_h,y_w,window_h,window_w,stride_h,stride_w,pad_top,pad_left,ceil_mode   |

#### 建模精度异常分析（数据采集正常且建模精度不达标必写，其余请忽略）
部分算子由于底层实现的复杂性，开发者难以建立达到目标精度的性能评估模型。开发者在与众智接口人协商确认后，可以启动异常验收流程。本章节为此异常验收流程的主要交付内容之一。
+ 说明建模过程中尝试过的各个模型（包括超参数）及使用的算子特征
+ 说明各模型的训练及测试精度

| 建模场景                                        | 训练集大小 | 80_percentage_error(取k折交叉验证的平均值) | 测试集大小 | 80_percentage_error | 模型超参数 | 算子特征                                                                                       |
|---------------------------------------------|-------|---------------------------------|-------|---------------------|------|--------------------------------------------------------------------------------------------|
| mode=1，global_pooling=0，xgboost模型           | 4000  | 0.283123                        | 2000  | 0.263006            |  learning_rate=0.11,n_estimators=350,max_depth=6,subsample=0.9,colsample_bytree=1   | flops,N,Cin,x_h,x_w,y_h,y_w,window_h,window_w,stride_h,stride_w,pad_top,pad_left,ceil_mode |
| mode=1，global_pooling=0，RandomForest模型      | 4000  | 0.507574                        | 2000  | 0.481651           |  n_estimators=350,max_depth=6   | flops,N,Cin,x_h,x_w,y_h,y_w,window_h,window_w,stride_h,stride_w,pad_top,pad_left,ceil_mode  |
| mode=1，global_pooling=0 ，GradientBoosting模型 | 4000  | 0.2670672                       | 2000  | 0.273444            |  learning_rate=0.11,n_estimators=350,max_depth=6,subsample=0.9,colsample_bytree=1   | flops,N,Cin,x_h,x_w,y_h,y_w,window_h,window_w,stride_h,stride_w,pad_top,pad_left,ceil_mode   |
| mode=1，global_pooling=0 ，ExtraTrees模型        | 4000  | 0.669689                        | 2000  |  0.648491                   |          n_estimators=350,max_depth=6                               |      flops,N,Cin,x_h,x_w,y_h,y_w,window_h,window_w,stride_h,stride_w,pad_top,pad_left,ceil_mode    |



### 测试结论
针对建模结果，在以下三条结论中选择对应的那一条
+ mode=1且global_pooling=0的场景下，算子性能评估模型训练成功，但建模精度不达标
+ 其余场景下算子性能评估模型训练成功，且建模精度达标。