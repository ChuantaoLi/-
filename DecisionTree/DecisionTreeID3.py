import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def split(dataset, feature_index, threshold):
    '''分裂数据'''
    dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])  # 左子树
    dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])  # 右子树
    return dataset_left, dataset_right  # 返回左右子树


def entropy(y):
    '''计算熵'''
    class_labels = np.unique(y)
    entropy = 0
    for cls in class_labels:
        p_cls = len(y[y == cls]) / len(y)
        entropy += -p_cls * np.log2(p_cls)
    return entropy


def gini_index(y):
    '''计算基尼指数'''
    class_labels = np.unique(y)
    gini = 0
    for cls in class_labels:
        p_cls = len(y[y == cls]) / len(y)
        gini += p_cls ** 2
    return 1 - gini


def information_gain(parent, l_child, r_child):
    '''计算信息增益'''
    weight_l = len(l_child) / len(parent)
    weight_r = len(r_child) / len(parent)
    gain = gini_index(parent) - (weight_l * gini_index(l_child) + weight_r * gini_index(r_child))
    return gain


def calculate_leaf_value(Y):
    '''计算叶节点的值'''
    Y = list(Y)
    return max(Y, key=Y.count)


def get_best_split(dataset, num_features):
    '''寻找最佳分裂方式'''
    best_split = {}
    max_info_gain = -float("inf")

    for feature_index in range(num_features):  # 遍历所有特征
        feature_values = dataset[:, feature_index]  # 获取特征值
        possible_thresholds = np.unique(feature_values)  # 获取特征值的唯一值作为可能的分裂点
        for threshold in possible_thresholds:  # 遍历所有可能的分裂点
            dataset_left, dataset_right = split(dataset, feature_index, threshold)  # 小于等于阈值的数据被划分为左子树，大于阈值的数据被划分为右子树
            if len(dataset_left) > 0 and len(dataset_right) > 0:  # 如果左右子树都不为空
                y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                curr_info_gain = information_gain(y, left_y, right_y)  # 计算信息增益
                if curr_info_gain > max_info_gain:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["dataset_left"] = dataset_left
                    best_split["dataset_right"] = dataset_right
                    best_split["info_gain"] = curr_info_gain
                    max_info_gain = curr_info_gain

    return best_split


def build_tree(dataset, min_samples_split, max_depth, curr_depth=0):
    '''构建决策树'''
    X, Y = dataset[:, :-1], dataset[:, -1]  # 分离训练集的特征和目标变量
    num_samples, num_features = np.shape(X)  # 获取样本数和特征数
    if num_samples >= min_samples_split and curr_depth <= max_depth:  # 要满足最小分裂样本数和最大深度的限定条件才能建树
        best_split = get_best_split(dataset, num_features)  # 获取最佳分裂方式，返回一个字典
        if best_split["info_gain"] > 0:  # 递归进行左右子树的构建。信息增益衡量了使用某个特征进行分割后，数据集纯度的提升程度
            left_subtree = build_tree(best_split["dataset_left"], min_samples_split, max_depth, curr_depth + 1)
            right_subtree = build_tree(best_split["dataset_right"], min_samples_split, max_depth, curr_depth + 1)
            return {"feature_index": best_split["feature_index"], "threshold": best_split["threshold"],
                    "left": left_subtree, "right": right_subtree, "info_gain": best_split["info_gain"]}

    leaf_value = calculate_leaf_value(Y)  # 既然不往下分裂子树，那么就是一个叶子节点，即类别，使用最多的类别作为该叶子节点的类别
    return {"value": leaf_value}


def print_tree(tree, col_names, indent=" "):
    '''打印树，使用特征名称，限制小数点后四位'''
    if "value" in tree:
        print(tree["value"])
    else:
        feature_name = col_names[tree["feature_index"]]  # 获取特征名称
        # 转换为浮点数并限制小数点后四位
        threshold = f"{float(tree['threshold']):.1f}"
        info_gain = f"{float(tree['info_gain']):.4f}"
        print(f"{feature_name} <= {threshold} ? {info_gain}")
        print(f"{indent}left:", end="")
        print_tree(tree["left"], col_names, indent + indent)
        print(f"{indent}right:", end="")
        print_tree(tree["right"], col_names, indent + indent)


def make_prediction(x, tree):
    '''预测单个数据点'''
    if "value" in tree:
        return tree["value"]
    feature_val = x[tree["feature_index"]]
    if feature_val <= tree["threshold"]:
        return make_prediction(x, tree["left"])
    else:
        return make_prediction(x, tree["right"])


def predict(X, tree):
    '''预测新数据集'''
    predictions = [make_prediction(x, tree) for x in X]
    return predictions


"""调用构建的决策树模型"""
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("iris.csv", header=None, names=col_names)
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=514)

# 训练决策树
dataset = np.concatenate((X_train, Y_train), axis=1)
tree = build_tree(dataset, min_samples_split=3, max_depth=5)  # 最小分裂样本数为3，最大深度为5
print_tree(tree, col_names)  # 打印决策树

# 预测
Y_pred = predict(X_test, tree)

# 评估准确率
print(f'{accuracy_score(Y_test, Y_pred):.4f}')
