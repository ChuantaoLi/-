import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def stump_classify(data_matrix, dim, thresh_val, ineq, left_class, right_class):
    """ 多分类决策树桩 """
    mask = (data_matrix[:, dim] <= thresh_val) if ineq == "lt" else (data_matrix[:, dim] > thresh_val)
    return np.where(mask, left_class, right_class).reshape(-1, 1)


def build_stump(data_arr, class_labels, D):
    """
    修正后的多分类决策树桩构建
    :param data_arr:    数据数组
    :param class_labels:    类别标签
    :param D:   权重向量
    :return:    最佳决策树桩，最小误差，最佳类别估计
    """
    data_matrix = np.array(data_arr)  # 转换为数组
    label_mat = np.array(class_labels).reshape(-1, 1)  # 保证列向量
    m, n = data_matrix.shape  # 样本数，特征数
    classes = np.unique(class_labels)  # 类别取值列表
    best_stump = {}  # 最佳决策树桩
    best_class_est = np.zeros((m, 1))  # 最佳类别估计
    min_error = np.inf  # 最小误差

    for i in range(n):  # 遍历所有特征
        feature_values = np.unique(data_matrix[:, i])  # 获取当前特征的所有唯一值作为可能的阈值
        for thresh_val in feature_values:  # 遍历每个可能阈值
            for inequal in ["lt", "gt"]:  # 遍历大于和小于等于两种不等式的阈值分割方式
                if inequal == "lt":
                    mask = data_matrix[:, i] <= thresh_val
                else:
                    mask = data_matrix[:, i] > thresh_val

                # 分别对不等式左右两边的数据进行处理
                # 转换为数组并展平维度
                left_weights = D[mask].flatten()  # 权重
                left_labels = label_mat[mask].flatten()  # 类别

                if len(left_labels) == 0:
                    left_class = -1  # 左侧无数据，设置异常
                else:
                    # 使用加权计数选择类别
                    class_scores = [np.sum(left_weights * (left_labels == c)) for c in classes]  # 对于每个类别，计算左侧区域中属于该类别的样本的权重之和
                    left_class = classes[np.argmax(class_scores)]  # 选择权重之和最大的类别作为左侧区域的类别，这和二分类有区别，二分类是直接选择错误率最小的类别

                # 对右侧区域进行相同处理
                right_weights = D[~mask].flatten()
                right_labels = label_mat[~mask].flatten()

                if len(right_labels) == 0:
                    right_class = -1
                else:
                    class_scores = [np.sum(right_weights * (right_labels == c)) for c in classes]
                    right_class = classes[np.argmax(class_scores)]

                # 跳过无效分割
                if left_class == -1 or right_class == -1:
                    continue

                # 生成预测结果
                predicted_vals = np.where(mask.reshape(-1, 1), left_class, right_class)  # 预测结果
                err_arr = (predicted_vals != label_mat).astype(float)  # 错误率数组
                weighted_error = np.dot(D.flatten(), err_arr.flatten())  # 显式转换为标量

                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump = {
                        "dim": i,
                        "thresh": thresh_val,
                        "ineq": inequal,
                        "left_class": left_class,
                        "right_class": right_class
                    }
    return best_stump, min_error, best_class_est


def ada_boost_train_ds(data_arr, class_labels, num_it):
    """
    训练函数
    :param data_arr: 数据数组
    :param class_labels: 类别标签
    :param num_it: 迭代次数
    """
    weak_class_arr = []  # 弱分类器列表
    m = data_arr.shape[0]  # 样本数
    K = len(np.unique(class_labels))  # 类别数量
    D = np.ones(m) / m  # 使用数组代替矩阵

    for _ in range(num_it):  # 迭代次数
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)  # 构建决策树桩

        # 添加安全阈值
        epsilon = 1e-10
        error = np.clip(error, epsilon, 1 - epsilon)

        # SAMME算法
        alpha = np.log((1.0 - error) / error) + np.log(K - 1)

        # 更新权重
        correct = (class_est.flatten() == class_labels)
        expon = np.where(correct, -alpha / (K - 1), alpha)
        D *= np.exp(expon)
        D /= D.sum()

        best_stump["alpha"] = alpha
        weak_class_arr.append(best_stump)

    return weak_class_arr


def ada_classify(data_to_class, classifier_arr):
    """ 多分类预测 """
    data_matrix = np.mat(data_to_class)
    m = data_matrix.shape[0]
    K = max([c["left_class"] for c in classifier_arr] + [c["right_class"] for c in classifier_arr]) + 1
    score_matrix = np.zeros((m, K))

    for classifier in classifier_arr:
        preds = stump_classify(
            data_matrix,
            classifier["dim"],
            classifier["thresh"],
            classifier["ineq"],
            classifier["left_class"],
            classifier["right_class"]
        )
        for i in range(m):
            score_matrix[i, int(preds[i, 0])] += classifier["alpha"]

    return np.argmax(score_matrix, axis=1)


def calculate_metrics(test_labels, predictions):
    """ 多分类评估指标 """
    try:
        auc = roc_auc_score(test_labels, predictions, multi_class='ovr')
    except:
        auc = 0.5  # 当所有预测为同一类时的默认值

    # 计算G-Mean（各类别召回率的几何平均）
    report = classification_report(test_labels, predictions, output_dict=True)
    recalls = [report[str(i)]['recall'] for i in range(len(np.unique(test_labels)))]
    g_mean = np.prod(recalls) ** (1 / len(recalls))

    return auc, g_mean


data = pd.read_csv("iris.csv")
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# 对 Y 进行标签编码
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

train_data, test_data, train_labels, test_labels = train_test_split(X, Y_encoded, test_size=0.2, random_state=514)

classifier_list = ada_boost_train_ds(train_data, train_labels, 10)

# 预测
predictions = ada_classify(test_data, classifier_list)

# 评估
auc, g_mean = calculate_metrics(test_labels, predictions)
print(f"OVR AUC: {auc:.4f}")
print(f"G-Mean: {g_mean:.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, predictions, digits=4))
cm = confusion_matrix(test_labels, predictions)
print("\nConfusion Matrix:")
print(cm)
