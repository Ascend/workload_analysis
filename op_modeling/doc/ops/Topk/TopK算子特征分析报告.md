# TopK算子特征分析报告

## 算子说明
**在此处说明算子功能**
+ 说明算子计算的方式（以公式等形式展现）
  + 查找最后一个维度中最大的k个元素的值和索引

+ 说明算子约束（如算子输入输出取值范围，内存限制等）
  + x：1D或更高维度，Size(x_shape) > 32768时，x的shape[-1] >= 16
  + k：0D，在1到x的shape[-1]范围内取值
  + sorted、largest、dim：仅能为默认值
  + values、indices：values.shape = indices.shape = input.shape[:-1] + [k]


## 特征选择
**说明特征选择的理由，包括但不限于：**
+ x_size：x的Size，输入数据的复杂程度是影响性能的关键，重要性0.499
+ x_last：x最后一维的Size，算子每次在最后一维内选取k个数字，重要性0.215
+ values_size：输出数据的Size，重要性0.170
+ x_front: 除了最后一维x的Size，等价于求k的次数，重要性0.081
+ values_last：values最后一维的Size，等价于K值，重要性0.032