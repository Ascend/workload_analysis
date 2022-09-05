# PadV3算子特征分析报告
## 算子说明
**在此处说明算子功能**  
+ 说明算子计算的方式（以公式等形式展现）
  + 算子功能是对张量进行边缘填充
  + （1）输入：<br />x是待填充的张量<br />paddings指定在x每个维度上填充行数，paddings的大小为[n,2]，n是输入x的秩。[D,0]指定在x的第D维前面添加多少行，[D,1]指定在x的第D维后面添加多少行。
  + （2）mode默认值CONSTANT，constant_value默认值0，paddings_contiguous默认值true
  + （3）输出：输出张量y的大小计算公式：paddings[D,0] + tensor.dim_size(D) + paddings[D,1]
+ 说明算子约束（如算子输入输出取值范围，内存限制等）
  + 算子输入和输出的数据类型相同，均为float16和float
  + 输入x和输出y均符合昇腾NCHW维度限制
## 特征选择
**说明特征选择的理由，包括但不限于：**
+ 选择符合算子核心概念及算法的特征
+ 避免重复且无效的特征（通过特征重要性等手段识别）
   + 算子输入x和输出y的大小对算子执行速度有很大关系
   + PadV3算子支持下float和float16两种数据类型，所以dtype也是特征
   + 最终选取特征如下：
   + （1）x，x_front，x_behind：与算子输入有关
   + （2）y，y_front，y_behind：与算子输出有关
   + （3）y_x：与张量扩充大小有关
   + （4）is_float，is_float16，is_y_float，is_y_float16：与算子数据类型有关