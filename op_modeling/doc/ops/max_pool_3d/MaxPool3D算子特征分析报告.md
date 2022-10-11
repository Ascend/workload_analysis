# MaxPool3D算子特征分析报告

## 算子说明
**在此处说明算子功能**
+ 说明算子计算的方式（以公式等形式展现）
    + MaxPool3D算子根据参数ksize决定本次池化的感受野，strides参数决定窗口的滑动步长，padding参数表示池化填充，它是一个字符串，可以是"VALID"算法或者"SAME"算法。pads参数对原输入进行填充，结合步长参数stride对inputs进行池化
    + 每ksize大小的inputs特征值通过平均处理转为outputs的一个特征值，结合ksize和strides滑动实现对整个inputs的池化操作
    
+ 说明算子约束（如算子输入输出取值范围，内存限制等）
  + ksize[0] = 1 or ksize[4] = 1, ksize[1] * ksize[2] * ksize[3] <= 255.
  + strides[0] = 1 or strides[4] = 1, strides[1] <= 63, strides[1] >= 1, strides[2] <= 63, strides[2] >= 1, strides[3] <= 63, strides[3] >= 1.

## 特征选择
**说明特征选择的理由，包括但不限于：**
+ 选择符合算子核心概念及算法的特征
  + flops: MaxPool3D算子可以根据output的shape推算出算子的计算量级。flops = N * C * k_d * k_h * k_w * Output_d * Output_h * Output_w(特征说明：N:批次的数量，C:通道个数，k_d:窗口的深，k_h:窗口的高，k_w:窗口的宽，Output_d:输出特征图的深，Output_h:输出特征图的高，Output_w:输出特征图的宽)
  + N：池化批次的数量，该特征影响算子的运算次数。
  + Cin：特征图的通道数量
  + x_size：inputs的Size，输入数据的复杂程度是影响性能的关键
  + k_size：kernel的size，滑动窗口的大小
  + s_size：stride的size，步长的大小
  + x_d：inputs特征图的深，决定要处理特征图的计算量
  + x_h：inputs特征图的高，决定要处理特征图的计算量
  + x_w：inputs特征图的宽，决定要处理特征图的计算量
  + y_d：outputs特征图的深，体现窗口横向滑动的次数
  + y_h：outputs特征图的高，体现窗口横向滑动的次数
  + y_w：outputs特征图的宽，体现窗口纵向滑动的次数
  + k_d：滑动窗口的深，该特征影响算子的运算次数
  + k_h：滑动窗口的高，该特征影响算子的运算次数
  + k_h：滑动窗口的宽，该特征影响算子的运算次数
  + s_d：深度方向步长，该特征可以决定滑动窗口滑动的次数，从而影响算子计算数量
  + s_h：纵向步长，该特征可以决定滑动窗口滑动的次数，从而影响算子计算数量
  + s_w：横向步长，该特征可以决定滑动窗口滑动的次数，从而影响算子计算数量
  + y_size：outputs的Size
  + ceil_mode：“天花板”或者“地板”模式的选择，当剩余特征值不足window时，ceil_mode起到区分的作用