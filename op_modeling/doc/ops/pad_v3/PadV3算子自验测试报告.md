### 测试目标
+ 验证PadV3算子建模的训练过程及推理结果。
## 模型功能验证
### 执行入口脚本
`python  padv3/builder.py`

### 结果分析
#### 样本分布（数据采集正常必写，异常请忽略）
在此处说明数据采集的结果（以表格的形式呈现）
+ 说明采集的样本总量
+ 展现采集的数据类型，格式及形状的分布情况
+ 表格内容要能体现样本分布的广泛性

参考的表格如下:
| 变量 | x | paddings | constant_values | mode | paddings_contiguous | y |
|----|--------|--------|-----|-------|-------|-----|
| 范围 | datatype: **float16,float** ,format:  **NCHW**  ,shape: **4D**   | datatype:  **int32** ,  **int64**  ,format: **ND** ,shape: **2D**  |  datatype:  **int32** ,  **int64**  ,format: **NCHW** ,shape: **1D**    |   datatype: **string** , 取默认值constant  |   datatype: **bool** ,  取默认值true |   datatype: **float16,float** ,format: **NCHW** ,shape: **4D**  |
#### 建模结果（数据采集正常且建模精度达标必写，其余请忽略）
+ 说明最终使用的模型（包括超参数）及使用的算子特征
+ 说明模型的训练及测试精度
该章节内容以表格的形式呈现，参考的表格如下，其中不同建模场景指对算子样本空间的划分（如示例Add算子按datatype进行建模场景区分）

| 建模场景 | 训练集大小 | 80_percentage_error(取k折交叉验证的平均值) | 测试集大小 | 80_percentage_error | 模型超参数 |算子特征 |
|------|-------|----------------------------------|-------|---------------------|------|------|
| 场景1|  1200  |  0.070772|  240   |   0.069733   | learning_rate=0.1, n_estimators=500, max_depth=6,  subsample=0.9, colsample_bytree=0.9 |is_float,is_float16,is_y_float,is_y_float16,x,x_front,x_behind,y,y_front,y_behind,y_x |

### 测试结论
针对建模结果，在以下三条结论中选择对应的那一条
+ 算子性能评估模型训练成功，且建模精度达标。