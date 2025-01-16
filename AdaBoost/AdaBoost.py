import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report


def load_data(file_path):
    """
    加载数据文件，将数据和标签分别存储在两个列表中
    :param file_path: 文件的路径
    :return: 数据列表和标签列表
    """
    data = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            data.append([float(x) for x in parts[:-1]])
            labels.append(int(parts[-1]))
    return np.array(data), np.array(labels)


def stump_classify(data_matrix, dim, thresh_val, thresh_ineq):
    """
    基于单个特征和阈值对数据进行分类
    :param data_matrix: 数据矩阵
    :param dim: 特征维度
    :param thresh_val: 阈值
    :param thresh_ineq: 不等式符号，'lt' 表示小于等于，'gt' 表示大于
    :return: 分类结果
    """
    ret_array = np.ones((data_matrix.shape[0], 1))  # 首先创建一个与数据矩阵样本数相同的数组，并初始化默认类别为1
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dim] <= thresh_val] = -1.0  # 通过切片，将dim这一列中小于等于thresh_val的样本的类别设定为-1
    else:
        ret_array[data_matrix[:, dim] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, D):
    """
    构建一个弱分类器（决策树桩）
    :param data_arr: 数据数组
    :param class_labels: 类别标签
    :param D: 样本权重向量
    :return: 最佳决策树桩的信息（字典），最小错误率，最佳分类结果
    """
    data_matrix = np.mat(data_arr)  # 转换成numpy格式
    label_mat = np.mat(class_labels).T  # 转换成numpy格式以及转换成列向量
    m, n = np.shape(data_matrix)  # 获取矩阵的维度
    num_steps = 10.0  # 将特征的取值分为num_steps份
    best_stump = {}  # 用来存储最佳决策树桩的信息
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf
    for i in range(n):  # 外循环：遍历每一个特征
        range_min = data_matrix[:, i].min()  # 当前特征中，取值的最小值
        range_max = data_matrix[:, i].max()  # 当前特征中，取值的最大值
        step_size = (range_max - range_min) / num_steps  # 特征取值的间隔
        for j in range(-1, int(num_steps) + 1):  # 内循环：遍历所有的阈值
            for inequal in ['lt', 'gt']:  # 细分不等式划分方向
                thresh_val = (range_min + float(j) * step_size)  # 设定当前内循环的划分阈值thresh_val
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)  # 当前预测结果数组predicted_vals
                err_arr = np.mat(np.ones((m, 1)))  # 分类错误的记录数组，初始化全为1，表示分类错误
                err_arr[predicted_vals == label_mat] = 0  # 如果分类正确，就赋值为0
                weighted_error = D.T * err_arr  # 计算加权错误率
                if weighted_error < min_error:  # 如果说这个划分阈值的设定（所构建的决策树桩）的加权错误率小于之前的最小错误率
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()  # 拷贝当前分类结果
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est


def ada_boost_train_ds(data_arr, class_labels, num_it):
    """
    训练 AdaBoost 分类器，集成多个弱分类器
    :param data_arr: 数据数组
    :param class_labels: 类别标签
    :param num_it: 迭代次数
    :return: 弱分类器列表，每个弱分类器的权重列表
    """
    weak_class_arr = []  # 弱分类器列表，包含了每次迭代训练得到的弱分类的信息
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 初始化样本权重
    agg_class_est = np.mat(np.zeros((m, 1)))  # 存储多个弱分类器的加权预测结果
    alpha_list = []  # 存储每个弱分类器的权重
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
        # 返回的best_stump表示当前迭代的最佳决策树桩（包括最优特征、阈值和不等式方向）
        # error表示当前弱分类器的加权错误率
        # class_est表示当前弱分类器对样本的预测结果
        alpha = float(0.5 * np.log((1.0 - error) / max(error)))  # 当前弱分类器的权重
        alpha_list.append(alpha)
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)  # 权重更新公式的指数部分，multiply表示对应元素相乘
        D = np.multiply(D, np.exp(expon))  # 权重更新公式的分子部分
        D = D / D.sum()  # 除以归一化因子，得到最终的权重更新公式
        agg_class_est += alpha * class_est  # 当前所有弱分类器加权后的预测结果
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T,
                                 np.ones((m, 1)))  # 比较当前加权后的强分类器的预测结果，sign表示对正数元素返回1，负数元素返回-1
        error_rate = agg_errors.sum() / m  # 计算当前强分类器的分类错误比例
    return weak_class_arr, alpha_list  # weak_class_arr包含了每次迭代得到的弱分类器的信息，alpha_list包含了每个弱分类器的权重


def ada_classify(data_to_class, classifier_arr, alpha_list):
    """
    使用 AdaBoost 分类器进行分类
    :param data_to_class: 待分类的数据
    :param classifier_arr: 弱分类器列表
    :param alpha_list: 弱分类器的权重列表
    :return: 分类结果
    """
    data_matrix = np.mat(data_to_class)
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))  # 用来存储每个样本的累积分类结果
    for i in range(len(classifier_arr)):  # 遍历每一个训练好的弱分类器
        class_est = stump_classify(data_matrix, classifier_arr[i]['dim'], classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])
        agg_class_est += alpha_list[i] * class_est
    return np.sign(agg_class_est)


def calculate_metrics(test_labels, predictions):
    """
    计算 AVG-AUC、G-MEAN 和 RECALL
    :param test_labels: 测试集的真实标签
    :param predictions: 测试集的预测结果
    :return: AVG-AUC、G-MEAN 和 RECALL
    """
    # 将预测结果转换为一维数组
    predictions = np.array(predictions).flatten()
    # 计算 AUC
    auc = roc_auc_score(test_labels, predictions)
    # 计算 G-MEAN
    tp = np.sum((predictions == 1) & (test_labels == 1))
    tn = np.sum((predictions == -1) & (test_labels == -1))
    fp = np.sum((predictions == 1) & (test_labels == -1))
    fn = np.sum((predictions == -1) & (test_labels == 1))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    g_mean = np.sqrt(sensitivity * specificity)
    return auc, g_mean


if __name__ == "__main__":
    train_data, train_labels = load_data('haberman_train.csv')
    test_data, test_labels = load_data('haberman_test.csv')
    # 训练 AdaBoost 分类器
    classifier_list, alpha_list = ada_boost_train_ds(train_data, train_labels, 500)
    # 对测试集进行分类
    predictions = ada_classify(test_data, classifier_list, alpha_list)
    # 计算准确率
    correct = np.sum(predictions == np.mat(test_labels).T)
    accuracy = correct / len(test_labels)
    print(f"Accuracy: {accuracy}")
    # 计算 AVG-AUC、G-MEAN 和 RECALL
    auc, g_mean = calculate_metrics(test_labels, predictions)
    print(f"AVG-AUC: {auc}")
    print(f"G-MEAN: {g_mean}")
    print("\nClassification Report:")
    predictions = pd.DataFrame(predictions)
    print(classification_report(test_labels, predictions))
