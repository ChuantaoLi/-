## AdaBoost
算法流程学习参考：[【十分钟 机器学习 系列课程】 讲义（55）:AdaBoost例题讲解 - 简博士的文章 - 知乎](https://zhuanlan.zhihu.com/p/552996396)

测试数据集来源：[Haberman (Imbalanced) data set](https://sci2s.ugr.es/keel/dataset.php?cod=157)

以`test_size=0.3`, `random_state=42`划分训练集和测试集，迭代500轮后输出的评估指标为：

Accuracy: 0.6989247311827957

AVG-AUC: 0.543915040183697

G-MEAN: 0.4149889683565515

Classification Report:
              precision    recall  f1-score   support

          -1       0.74      0.90      0.81        67
           1       0.42      0.19      0.26        26

    accuracy                           0.70        93
   macro avg       0.58      0.54      0.54        93
weighted avg       0.65      0.70      0.66        93
