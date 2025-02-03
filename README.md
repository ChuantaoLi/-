## 二分类AdaBoost
算法流程学习参考：[【十分钟 机器学习 系列课程】 讲义（55）:AdaBoost例题讲解 - 简博士的文章 - 知乎](https://zhuanlan.zhihu.com/p/552996396)

测试数据集来源：[Haberman (Imbalanced) data set](https://sci2s.ugr.es/keel/dataset.php?cod=157)

以`test_size=0.3`, `random_state=42`划分训练集和测试集，迭代500轮后输出的评估指标为：

Accuracy: 0.6989247311827957

AVG-AUC: 0.543915040183697

G-MEAN: 0.4149889683565515

Classification Report:
```
              precision    recall  f1-score   support

          -1       0.74      0.90      0.81        67
           1       0.42      0.19      0.26        26

    accuracy                           0.70        93
   macro avg       0.58      0.54      0.54        93
weighted avg       0.65      0.70      0.66        93
```

## 多分类AdaBoost
以鸢尾花iris.csv数据集为例，安装`test_size=0.2, random_state=514`划分训练集和测试集，迭代10轮后输出的评估指标为：
```
OVR AUC: 0.5000
G-Mean: 0.9086

Classification Report:
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000        10
           1     0.8571    1.0000    0.9231        12
           2     1.0000    0.7500    0.8571         8

    accuracy                         0.9333        30
   macro avg     0.9524    0.9167    0.9267        30
weighted avg     0.9429    0.9333    0.9311        30


Confusion Matrix:
[[10  0  0]
 [ 0 12  0]
 [ 0  2  6]]
```
