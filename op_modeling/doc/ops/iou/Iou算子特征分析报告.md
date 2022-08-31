# 算子特征分析报告

## 算子说明
### 算子计算方法

$$
IOU=\frac {Area of Overlap}{Area of Union}
$$

$$
IOF=\frac {Area of Overlap}{Area of Ground Truth}
$$

### 算子参数

- **mode**（string）- 指定计算方法，现支持’iou’(intersection over union)或’iof’(intersection over foreground)模式。默认值：’iou’。

### 算子输入

- **anchor_boxes**（Tensor） - 预测区域，shape为(N, 4)的Tensor。”N”表示预测区域的数量，”4”表示”x0”、”y0”、”x1”和”y1”。数据类型为float16或float32。
- **gt_boxes**（Tensor）- 真实区域，shape为(M, 4)的Tensor。”M”表示地面真实区域的数量，”4”表示”x0”、”y0”、”x1”和”y1”。数据类型为float16或float32。

### 算子输出

- IOU值的Tensor，shape为(M, N)的Tensor，数据类型与 anchor_boxes 的相同。

### 约束

- 在Ascend中，仅支持计算float16数据。

## 特征选择
- 算子执行的时间通常与算子的FLOPS相关，所以选择anchor_boxes的数据规模、gt_boxes的数据规模和overlap的数据规模。
- 虽然支持采集float类型数据，但算子约束计算类型仅支持float16数据，故不考虑数据类型特征。