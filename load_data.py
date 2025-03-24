import os

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import cebra_code
import dim_handle
import eeg_handle
from cebra import CEBRA

import lstm_handle
import ml_handle
import report_handle

print(torch.__version__)  # 查看 PyTorch 版本
print(torch.version.cuda)  # 查看 PyTorch 编译时的 CUDA 版本
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA 设备数量:", torch.cuda.device_count())
    print("当前 CUDA 设备:", torch.cuda.get_device_name(0))
    print("CUDA 版本:", torch.version.cuda)

dataDirDev = "." + os.sep + "nasa_data"
modelDirDev = "." + os.sep + "model_eval"

# 检查 nasa_data 文件夹是否存在
if not os.path.exists(dataDirDev):
    print(f"Error: Directory '{dataDirDev}' does not exist.")
    exit(1)

# 初始化总 DataFrame
ca_total_df = pd.DataFrame()
da_total_df = pd.DataFrame()
ss_total_df = pd.DataFrame()
loft_total_df = pd.DataFrame()

# 遍历 nasa_data 文件夹下的所有子文件夹
def cebra_handle(train_result_dfs, loft_result_dfs, p_num):
    cebra_model_name = modelDirDev + os.sep + 'cebra_time_delta' + p_num + '.cebra'
    cebra_model,train_data,behavior_labels = None,None,None
    if os.path.exists(cebra_model_name):
        cebra_model = CEBRA.load(cebra_model_name, weights_only=False)
        print("模型加载成功！")
        print(f"文件 {cebra_model_name} 存在。")
    else:
        cebra_model,train_data,behavior_labels = cebra_code.cebra_model(train_result_dfs)
        try:
            cebra_model.save(cebra_model_name)
        except Exception as e:
            print("save error")

    # cebra_code.plot_cebra(train_result_dfs, p_num, cebra_model, 700, 'baseline')
    cebra_code.plot_cebra(loft_result_dfs, p_num, cebra_model, 700, 'loft')
    return cebra_model,train_data,behavior_labels


def model_handle(loft_df, param , p_num , window_size= 10):
    report_model_name = report_handle.reportDirDev + os.sep + param + '_' + p_num + '_report.csv'
    if os.path.exists(report_model_name):
        print(f"文件 {report_model_name} 存在。")
        return
    loft_x = loft_df.loc[:, eeg_handle.all_channel_names].values
    loft_y = loft_df.loc[:, "Event"].values
    X, y = eeg_handle.prepare_data(loft_x, loft_y)
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X, y = eeg_handle.create_sliding_window(X, y, window_size)
    # 平衡类别分布
    X, y = eeg_handle.balance_classes(X, y)

    X_train_split, X_predict_split, y_train_split, y_predict_split = train_test_split(
        X,
        y,
        test_size=0.25,  # 预测集占比 25%
        stratify=y,  # 保持类别分布一致（适用于分类任务）
        random_state=42  # 随机种子（确保可复现性）
    )

    lstm_model = eeg_handle.lstm(X_train_split, y_train_split)
    # 进行预测
    y_pred_prob = lstm_model.predict(X_predict_split)  # 预测概率
    y_pred = np.argmax(y_pred_prob, axis=1)  # 将概率转换为类别标签
    y_true = np.argmax(y_predict_split, axis=1)  # 将概率转换为类别标签
    accuracy, f1, report = report_handle.get_report(y_true, y_pred)
    report_handle.save(report,report_model_name)


def init_data(cebra_loft_embedding, loft_df):
    X_train = cebra_loft_embedding
    y_train = loft_df.loc[:, 'Event']
    y_list = np.array(y_train)
    # 确保 y_list 是 NumPy 数组
    y_list = np.where(y_list == 5, 3, y_list)  # 将 5 映射为 3

    under_sampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = under_sampler.fit_resample(X_train, y_list)

    X_train_split, X_predict_split, y_train_split, y_predict_split = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.25,  # 预测集占比 25%
        stratify=y_resampled,  # 保持类别分布一致（适用于分类任务）
        random_state=42  # 随机种子（确保可复现性）
    )
    return  X_train_split, X_predict_split, y_train_split, y_predict_split


for root, dirs, files in os.walk(dataDirDev):
    # 过滤掉根目录，只处理子文件夹
    if root != dataDirDev:
        # 打印当前子文件夹路径
        print(f"Processing folder: {root}")
        p_num = os.path.basename(root)
        # 获取当前子文件夹中的文件列表
        file_list = os.listdir(root)

        # 检查是否有四个文件
        if len(file_list) == 4:
            ca_df = None
            da_df = None
            loft_df = None
            ss_df = None
            for file_name in file_list:
                file_path = os.path.join(root, file_name)
                print(f"  Reading file: {file_path}")
                if file_name.endswith('CA.csv'):
                    try:
                        ca_df = pd.read_csv(file_path)
                        print("  Successfully read CA file as DataFrame.")
                    except Exception as e:
                        print(f"  Error reading CA file: {e}")
                elif file_name.endswith('DA.csv'):
                    try:
                        da_df = pd.read_csv(file_path)
                        print("  Successfully read DA file as DataFrame.")
                    except Exception as e:
                        print(f"  Error reading DA file: {e}")
                elif file_name.endswith('LOFT.csv'):
                    try:
                        loft_df = pd.read_csv(file_path)
                        print("  Successfully read LOFT file as DataFrame.")
                    except Exception as e:
                        print(f"  Error reading LOFT file: {e}")
                elif file_name.endswith('SS.csv'):
                    try:
                        ss_df = pd.read_csv(file_path)
                        print("  Successfully read SS file as DataFrame.")
                    except Exception as e:
                        print(f"  Error reading SS file: {e}")
                else:
                    print("  File does not match the expected suffixes.")

            ca_df, da_df, ss_df, loft_df = eeg_handle.handle_all(ca_df, da_df, ss_df, loft_df)

            # 使用 concat 将新的 DataFrame 追加到总 DataFrame 中
            ca_total_df = pd.concat([ca_total_df, ca_df], ignore_index=True)
            da_total_df = pd.concat([da_total_df, da_df], ignore_index=True)
            ss_total_df = pd.concat([ss_total_df, ss_df], ignore_index=True)
            loft_total_df = pd.concat([loft_total_df, loft_df], ignore_index=True)

            train_result_dfs, loft_result_dfs = cebra_code.init_data(ca_df, da_df, loft_df, ss_df)
            cebra_model,train_data,behavior_labels = cebra_handle(train_result_dfs, loft_result_dfs, p_num)

            # cebra_loft_embedding = cebra_model.transform(loft_df.loc[:, eeg_handle.eeg_channel_names])
            # X_cebra_train_split, X_cebra_predict_split, y_train_split, y_predict_split = init_data(cebra_loft_embedding , loft_df)
            #
            # ml_handle.process(X_cebra_train_split, X_cebra_predict_split, y_train_split, y_predict_split, "cebra", p_num)

            # X_train_split = dim_handle.get_tsne_loft_embedding(X_train_split)
            # X_predict_split = dim_handle.get_tsne_loft_embedding(X_predict_split)
            # ml_handle.process(X_train_split, X_predict_split, y_train_split, y_predict_split, "cebra_tsne", p_num)

            # X_train_split = dim_handle.get_umap_loft_embedding(X_train_split)
            # X_predict_split = dim_handle.get_umap_loft_embedding(X_predict_split)
            # ml_handle.process(X_train_split, X_predict_split, y_train_split, y_predict_split, "cebra_umap", p_num)

            #lstm_handle.process3(loft_df, cebra_model, "cebra_ds-umap-lstm", p_num)
            window_size = 50
            model_handle(loft_df, "none-lstm", p_num , window_size)
            lstm_handle.process2(loft_df, cebra_model, "cebra-ds-lstm", p_num , window_size)
            lstm_handle.process(loft_df, cebra_model,"cebra_ds-lstm-attention" , p_num , window_size)













