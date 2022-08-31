# 算子特征分析报告

## 算子说明
**在此处说明算子功能**
+ 说明算子计算的方式（以公式等形式展现）
    + Pooling算子首先根据参数global_pooling确实是否采取全局池化；根据参数mode确定本次池化采取策略为最大池化或者平均池化
    + 根据参数window决定本次池化的感受野，pad参数对原输入进行填充，dilation参数决定感受野是否进行膨胀，结合步长参数stride对inputs进行池化
    + 每window大小的inputs特征值通过最大或者平均处理转为outputs的一个特征值，结合stride和window滑动实现对整个inputs的池化操作
    
+ 说明算子约束（如算子输入输出取值范围，内存限制等）
  + window[0] * window[1] < 256
  + 1<=input_h<=4096,1<=input_w<=4096
  + If input tensor N is a prime number, it should be less than 65535

## 特征选择
**说明特征选择的理由，包括但不限于：**
+ 选择符合算子核心概念及算法的特征
  + flops: Pooling算子可以根据output的shape推算出算子的计算量级。flops = N * C * W_h * W_w * Output_h * Output_w(特征说明：N:批次的数量，C:通道个数，W_h:窗口的高， W_w:窗口的宽，Output_h:输出特征图的高，Output_w:输出特征图的宽)
  + N：池化批次的数量，该特征影响算子的运算次数。
  + Cin：特征图的通道数量
  + x_h：inputs特征图的高，决定要处理特征图的计算量
  + x_w：inputs特征图的宽，决定要处理特征图的计算量
  + y_h：outputs特征图的高，体现窗口横向滑动的次数
  + y_w：outputs特征图的宽，体现窗口纵向滑动的次数
  + window_h：滑动窗口的高，该特征影响算子的运算次数
  + window_h：滑动窗口的宽，该特征影响算子的运算次数
  + stride_h：纵向步长，该特征可以决定滑动窗口滑动的次数，从而影响算子计算数量
  + stride_w：横向步长，该特征可以决定滑动窗口滑动的次数，从而影响算子计算数量
  + pad_top：特征图top方向上的填充，一般情况下，pad_bottom与pad_top一致，该特征可能会对算子增加数量的运算
  + pad_left：特征图left方向上的填充，一般情况下，pad_right与pad_left一致，该特征可能会对算子增加数量的运算
  + ceil_mode：“天花板”或者“地板”模式的选择，当剩余特征值不足window时，ceil_mode起到区分的作用。
