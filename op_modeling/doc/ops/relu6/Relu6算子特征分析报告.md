# 算子特征分析报告

## 算子说明
**算子功能：以6为上限修正线性单元**

+ 算子计算公式：
  $$
  Y=min(max(0,x),6)
  $$
+ 算子输入：x：RealNumberType的Tensor。数据类型为float16，float，int32。
+ 算子输出：y：与x同策略的Tensor。数据类型为float16，float，int32。

## 特征选择
+ x：算子执行的时间与算子的FLOPS相关，即只与算子的输入数据x的大小有关，因此该算子的输入数据x是影响模型性能最重要的特征。