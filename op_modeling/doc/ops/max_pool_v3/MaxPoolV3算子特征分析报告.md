# MaxPoolV3算子特征分析报告

## 算子说明
**在此处说明算子功能**
+ 说明算子计算的方式（以公式等形式展现）

+ 说明算子约束（如算子输入输出取值范围，内存限制等）

  （1）输入：x，数据格式为NHWC

  （2）参数：滑窗大小ksize，步长strides

  （3）输出：

  SAME模式：

​				y_h_shape = math.ceil(x_h_shape / strides_h_value)

​				y_w_value = math.ceil(x_w_shape / strides_w_value)

​		VALID模式：

​				y_h_shape = math.ceil((x_h_shape - ksize_h_value + 1) / strides_h_value)

​				y_w_shape = math.ceil((x_w_shape - ksize_w_value + 1) / strides_w_value)

​		CALCULATED模式：

​				y_h_value = math.floor((x_shape[1] + (pad_top + pad_bottom) - ksize_h_value + ceil_mode_pad_h) / strides_h_value + 1)

​				y_w_value = math.floor((x_shape[2] +  (pad_left + pad_right) - ksize_w_value + ceil_mode_pad_w) / strides_w_value + 1)

​		GLOBAL_POOLING模式：

​				y_shape = [x_shape[0], 1, 1, x_shape[3]]

## 特征选择
**说明特征选择的理由，包括但不限于：**
+ 选择符合算子核心概念及算法的特征
+ 避免重复且无效的特征（通过特征重要性等手段识别）

（1）flops：MaxPool算子可以根据output的shape推算出算子的计算量级。flops=x_N ** x_C ** ksize_H ** ksize_W ** y_H * y_W

（2）不同模式也有区别：is_SAME，is_VALID，is_CALCULATED，is_global

（3）输入特征图的大小，是影响性能的关键：H_x和H_y

（4）输出特征图的打小，体现计算量：H_x和H_y，以及N_y和C_y

（5）滑动窗口的高和宽，影响覆盖的打小，影响计算的次数：H_ksize和W_ksize

（6）移动步长，影响窗口移动的次数，从而影响算子计算量：H_strides和W_strides

（7）SAME模式和VALID模式也对计算量有一定影响：is_SAME

（8）pads的大小也影响计算量