# import csv
#
# with open('haberman.dat', 'r') as dat_file:
#     with open('haberman.csv', 'w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         for row in dat_file:
#             row = [value.strip() for value in row.split('|')]
#             csv_writer.writerow(row)

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_dataset(file_path, test_ratio):
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    # 将数据集划分为训练集和测试集
    train_set, test_set = train_test_split(df, test_size=test_ratio, random_state=42)
    return train_set, test_set


def main():
    file_path = r'D:\Dynamically_Adaptive_Class_Balanced_XGBoost\haberman.xlsx'
    train_set, test_set = load_and_split_dataset(file_path, test_ratio=0.3)
    print("训练集的形状:", train_set.shape)
    print("测试集的形状:", test_set.shape)
    train_set = pd.DataFrame(train_set)
    test_set = pd.DataFrame(test_set)
    train_set.to_csv(r'haberman_train.csv', index=False)
    test_set.to_csv(r'haberman_test.csv', index=False)


if __name__ == "__main__":
    main()
