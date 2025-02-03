## 决策树ID3代码复现

决策树原理学习：[Decision Tree Classification Clearly Explained!](https://www.youtube.com/watch?v=ZVR2Way4nwQ)

代码参考：[ML_from_Scratch/decision tree classification.ipynb](https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20classification.ipynb)

以`test_size=0.2, random_state=514`划分训练集和测试集，得到的评估指标如下：
```
Classification Report:
                 precision    recall  f1-score   support

    Iris-setosa     1.0000    1.0000    1.0000        12
Iris-versicolor     0.9091    0.9091    0.9091        11
 Iris-virginica     0.8750    0.8750    0.8750         8

       accuracy                         0.9355        31
      macro avg     0.9280    0.9280    0.9280        31
   weighted avg     0.9355    0.9355    0.9355        31

[[12  0  0]
 [ 0 10  1]
 [ 0  1  7]]
```
建立的决策树如下：
```
petal_length <= 1.9 ? 0.3221
 left:Iris-setosa
 right:petal_length <= 4.7 ? 0.3807
  left:Iris-versicolor
  right:petal_width <= 1.6 ? 0.0595
    left:petal_length <= 4.9 ? 0.3000
        left:Iris-versicolor
        right:sepal_length <= 6.0 ? 0.1200
                left:Iris-versicolor
                right:Iris-virginica
    right:sepal_length <= 7.9 ? 0.0475
        left:petal_length <= 4.8 ? 0.0158
                left:sepal_length <= 5.9 ? 0.4444
                                left:Iris-versicolor
                                right:Iris-virginica
                right:Iris-virginica
        right:Species
```
