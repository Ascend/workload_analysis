# 算子特征分析报告

## 算子说明
**该算子为计算二元分类器损失的损失函数，公式为：**
+ reduction为"none"时：$ L = -W (Y \cdot log(X) + (1 - Y) \cdot log(1 - X)) $
+ reduction为"mean"时：$ L = mean(L) $
+ reduction为"sum"时：$ L = sum(L) $
+ 在如上公式中，X和Y的取值范围均为0-1，W为可选参数，X、Y、W形状相同，输出在reduction为"none"时形状与X相同，其他情况下为常数

## 特征选择
由上可知该算子有以下特征：X的大小，是否有W，L的大小，reduction的值  
此外该算子支持float和float16两种数据类型，故dtype也是一个特征  
通过观察可知，L的大小与reduction为重复特征，L的大小完全由reduction决定，故仅需保留reduction  
综上，最终选定的特征为X的大小，W是否给出，reduction，dtype四个