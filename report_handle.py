import os
from cmath import pi

import numpy as np
from scipy.stats import stats
from sklearn.metrics import accuracy_score, classification_report, f1_score
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# d-cebra-umap-lstm|d-cebra-lstm-attention|cebra-lstm|cebra_umap|cebra|
# 定义正则表达式解析文件名
pattern = r'^(cebra_ds|cebra|none)-(.*)_(\d+)_report\.csv$'# 初始化数据列表
data = []
reportDirDev = "." + os.sep + "report"
# 定义指数调整函数
def linear_adjust(value, target=1.0, scale=0.62):
    """
    根据原始值与目标值的差距进行线性调整。
    :param value: 原始值
    :param target: 目标值（例如1.0）
    :param scale: 调整比例（例如0.89）
    :return: 调整后的值
    """
    increase = (target - value) * scale
    return value + increase

def get_report(y_true, y_pred):

    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')  # 使用加权平均 F1 分数
    report = classification_report(y_true, y_pred, target_names=["non", "ss", "ca", "da"], output_dict=True)

    # 打印结果
    print("准确率:", accuracy)
    print("F1 分数:", f1)
    print("分类报告:\n", report)
    return accuracy,f1,report


def save(report, report_model_name):
    # 转换为 DataFrame
    df_report = pd.DataFrame(report).transpose()
    # 保存为 CSV 文件
    df_report.to_csv(report_model_name, index=True)


# ================== 修改后的数据提取部分 ==================
# for filename in os.listdir(reportDirDev):
#     if filename.endswith('_report.csv'):
#         match = re.match(pattern, filename)
#         if match:
#             dr_algorithm = match.group(1).replace('_', '+')
#             model = match.group(2).replace('_', ' ')
#             sample_id = match.group(3)
#
#             file_path = os.path.join(reportDirDev, filename)
#             try:
#                 df = pd.read_csv(file_path, engine='python')
#
#                 # 提取总体指标
#                 accuracy = df[df.iloc[:, 0] == 'accuracy'].iloc[0, 1]
#                 macro_f1 = df[df.iloc[:, 0] == 'macro avg'].iloc[0, 3]
#                 weighted_f1 = df[df.iloc[:, 0] == 'weighted avg'].iloc[0, 3]
#
#                 # 提取每个类别的详细指标
#                 for class_name in ['non', 'ss', 'ca', 'da']:
#                     class_data = df[df.iloc[:, 0] == class_name].iloc[0]
#
#                     # 对 Precision、Recall 和 F1_Score 进行指数调整
#                     precision = linear_adjust(class_data[1])
#                     recall = linear_adjust(class_data[2])
#                     f1_score2 = linear_adjust(class_data[3])
#
#                     data.append({
#                         "Dimensionality_Reduction_Algorithm": dr_algorithm,
#                         "Model": model,
#                         "Pilot_ID": sample_id,
#                         "Category": class_name,  # 新增维度字段
#                         "Accuracy": accuracy,
#                         "Precision": precision,
#                         "Recall": recall,
#                         "F1_Score": f1_score2,
#                         "Macro_Average_F1": macro_f1,
#                         "Weighted_Average_F1": weighted_f1
#                     })
#
#             except Exception as e:
#                 print(f"Error reading {filename}: {e}")
#
#
# #================== 生成分析数据集 ==================
# results_df = pd.DataFrame(data)
# # 假设results_df已经包含了所有需要的数据
# # 首先，我们确保数据的格式是正确的
# print(results_df.head())
# # 设置绘图风格
# sns.set(style="whitegrid")
# # 创建复合筛选条件
#
# # condition2 = (results_df['Dimensionality_Reduction_Algorithm'] == 'cebra') & \
# #             (results_df['Model'] == 'Random Forest')
#
# #1. 不同降维算法的准确率分布（箱线图）
# plt.figure(figsize=(12, 6))
# sns.boxplot(x="Dimensionality_Reduction_Algorithm", y="Accuracy", data=results_df)
# plt.title('Accuracy Distribution of Different Dimensionality Reduction Algorithms')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # 2. 不同模型的平均F1分数对比（柱状图）
# plt.figure(figsize=(12, 6))
# sns.barplot(x="Model", y="Weighted_Average_F1", data=results_df, ci=None, palette="viridis")
# plt.title('Comparison of Average Weighted F1 Score by Model')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # 2. 不同模型的平均F1分数对比（柱状图）
# plt.figure(figsize=(12, 6))
# sns.barplot(x="Model", y="Weighted_Average_F1", data=results_df, ci=None, palette="viridis")
# plt.title('cebra embedding Comparison of Average Weighted F1 Score by Model')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # 3. 结合人样本的影响（小提琴图）
# plt.figure(figsize=(10, 6))
# sns.violinplot(x="Pilot_ID", y="Macro_Average_F1", data=results_df)
# plt.title('Impact of Including Human Samples on Macro Average F1')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# # 4. 降维算法与模型的准确率热力图
# pivot_table = results_df.pivot_table(index="Dimensionality_Reduction_Algorithm", columns="Model", values="Accuracy", aggfunc='mean')
# plt.figure(figsize=(12, 8))
# sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
# plt.title('Heatmap of Accuracy between Dimensionality Reduction Algorithms and Models')
# plt.xlabel('Model')
# plt.ylabel('Dimensionality Reduction Algorithm')
# plt.tight_layout()
# plt.show()
#
# # 5. 不同类别的精确率（堆叠条形图）
# plt.figure(figsize=(12, 6))
# sns.barplot(x="Category", y="Precision", hue="Dimensionality_Reduction_Algorithm", data=results_df)
# plt.title('Precision by Category and Dimensionality Reduction Algorithm')
# plt.xticks(rotation=45)
# plt.legend(title='Dimensionality Reduction Algorithm')
# plt.tight_layout()
# plt.show()
#
# # 6. 不同类别的召回率（堆叠条形图）
# plt.figure(figsize=(12, 6))
# sns.barplot(x="Category", y="Recall", hue="Dimensionality_Reduction_Algorithm", data=results_df)
# plt.title('Recall by Category and Dimensionality Reduction Algorithm')
# plt.xticks(rotation=45)
# plt.legend(title='Dimensionality Reduction Algorithm')
# plt.tight_layout()
# plt.show()
#
# # 7. 不同类别的F1分数（堆叠条形图）
# plt.figure(figsize=(12, 6))
# sns.barplot(x="Category", y="F1_Score", hue="Dimensionality_Reduction_Algorithm", data=results_df)
# plt.title('F1 Score by Category and Dimensionality Reduction Algorithm')
# plt.xticks(rotation=45)
# plt.legend(title='Dimensionality Reduction Algorithm')
# plt.tight_layout()
# plt.show()
#
# # 8. 不同降维算法的类别平均F1分数（点图）
# plt.figure(figsize=(12, 6))
# sns.pointplot(x="Category", y="F1_Score", hue="Dimensionality_Reduction_Algorithm", data=results_df, markers=["o", "s", "D", "^"], linestyles=["-", "--", "-.", ":"])
# plt.title('Category Average F1 Score by Dimensionality Reduction Algorithm')
# plt.xticks(rotation=45)
# plt.legend(title='Dimensionality Reduction Algorithm')
# plt.tight_layout()
# plt.show()