import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import dim_handle
import eeg_handle
import report_handle


def process(loft_df,cebra_model, param, p_num,window_size=10):
    report_model_name = report_handle.reportDirDev + os.sep + param +'_'+ p_num + '_report.csv'
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

    # 将通道名称转换为整数索引
    channel_indices = [eeg_handle.all_channel_names.index(ch) for ch in eeg_handle.eeg_channel_names]
    cxt = X_train_split[:, window_size, channel_indices]  # 使用索引 9 获取最后一个时间步
    cxp = X_predict_split[:, window_size, channel_indices]
    cxt_embedding = cebra_model.transform(cxt)
    cxp_embedding = cebra_model.transform(cxp)

    # 数据标准化
    scaler = StandardScaler()
    cxt_embedding = scaler.fit_transform(cxt_embedding)
    cxp_embedding = scaler.fit_transform(cxp_embedding)

    lstm_model = eeg_handle.cebra_lstm_attention(X_train_split, cxt_embedding, y_train_split)
    # 进行预测
    y_pred_prob = lstm_model.predict([X_predict_split, cxp_embedding])  # 预测概率
    y_pred = np.argmax(y_pred_prob, axis=1)  # 将概率转换为类别标签
    y_true = np.argmax(y_predict_split, axis=1)  # 将概率转换为类别标签
    accuracy, f1, report = report_handle.get_report(y_true, y_pred)
    report_handle.save(report, report_model_name)
    return None


def process2(loft_df,cebra_model, param, p_num ,window_size= 10):
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

    # 将通道名称转换为整数索引
    channel_indices = [eeg_handle.all_channel_names.index(ch) for ch in eeg_handle.eeg_channel_names]
    cxt = X_train_split[:, window_size, channel_indices]  # 使用索引 9 获取最后一个时间步
    cxp = X_predict_split[:, window_size, channel_indices]
    cxt_embedding = cebra_model.transform(cxt)
    cxp_embedding = cebra_model.transform(cxp)

    # 数据标准化
    scaler = StandardScaler()
    cxt_embedding = scaler.fit_transform(cxt_embedding)
    cxp_embedding = scaler.fit_transform(cxp_embedding)

    lstm_model = eeg_handle.cebra_lstm(X_train_split, cxt_embedding, y_train_split)
    # 进行预测
    y_pred_prob = lstm_model.predict([X_predict_split, cxp_embedding])  # 预测概率
    y_pred = np.argmax(y_pred_prob, axis=1)  # 将概率转换为类别标签
    y_true = np.argmax(y_predict_split, axis=1)  # 将概率转换为类别标签

    accuracy, f1, report = report_handle.get_report(y_true, y_pred)
    report_handle.save(report, report_model_name)
    return None


def process3(loft_df, cebra_model, param, p_num):
    loft_x = loft_df.loc[:, eeg_handle.eeg_channel_names].values
    loft_y = loft_df.loc[:, "Event"].values
    X, y = eeg_handle.prepare_data(loft_x, loft_y)
    X, y = eeg_handle.create_sliding_window(X, y, window_size=10)
    # 平衡类别分布
    X, y = eeg_handle.balance_classes(X, y)
    cX = [dim_handle.get_umap_loft_embeddingV2(cebra_model.transform(X[i])) for i in range(len(X))]

    # under_sampler = RandomUnderSampler(random_state=42)
    # X_resampled, y_resampled = under_sampler.fit_resample(X, y)

    X_train_split, X_predict_split, y_train_split, y_predict_split = train_test_split(
        cX,
        y,
        test_size=0.25,  # 预测集占比 25%
        stratify=y,  # 保持类别分布一致（适用于分类任务）
        random_state=42  # 随机种子（确保可复现性）
    )
    X_train_split = np.array(X_train_split)
    X_predict_split = np.array(X_predict_split)
    lstm_model = eeg_handle.cebra_lstm_v3(X_train_split, y_train_split)
    # 进行预测
    y_pred_prob = lstm_model.predict(X_predict_split)  # 预测概率
    y_pred = np.argmax(y_pred_prob, axis=1)  # 将概率转换为类别标签
    y_true = np.argmax(y_predict_split, axis=1)  # 将概率转换为类别标签
    accuracy, f1, report = report_handle.get_report(y_true, y_pred)
    report_model_name = report_handle.reportDirDev + os.sep + param + '_' + p_num + '_report.csv'
    report_handle.save(report, report_model_name)

    return None