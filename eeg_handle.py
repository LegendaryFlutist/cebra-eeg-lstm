import os
import math
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, MultiHeadAttention, LayerNormalization, Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.src.layers import Bidirectional, BatchNormalization, GlobalAveragePooling1D
from keras.src.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import tensorflow as tf
import cebra_code
import mne_eeg_pretreatment
import report_handle

from ClassMetricsCallback import ClassMetricsCallback

eeg_channel_names = [
    'EEG_FP1', 'EEG_F7', 'EEG_F8', 'EEG_T4', 'EEG_T6',
    'EEG_T5', 'EEG_T3', 'EEG_FP2', 'EEG_O1', 'EEG_P3',
    'EEG_Pz', 'EEG_F3', 'EEG_Fz', 'EEG_F4', 'EEG_C4',
    'EEG_P4', 'EEG_POz', 'EEG_C3', 'EEG_Cz', 'EEG_O2'
]
all_channel_names = [
    'EEG_FP1', 'EEG_F7', 'EEG_F8', 'EEG_T4', 'EEG_T6',
    'EEG_T5', 'EEG_T3', 'EEG_FP2', 'EEG_O1', 'EEG_P3',
    'EEG_Pz', 'EEG_F3', 'EEG_Fz', 'EEG_F4', 'EEG_C4',
    'EEG_P4', 'EEG_POz', 'EEG_C3', 'EEG_Cz', 'EEG_O2',
    'ECG', 'R', 'GSR',
                                             ]


modelDirDev = "." + os.sep + "model_eval"

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)
        self.out_dense = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        query = self.query_dense(x)  # (batch_size, seq_len, d_model)
        key = self.key_dense(x)
        value = self.value_dense(x)

        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len, d_head)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        score = tf.matmul(query, key, transpose_b=True)  # (batch_size, num_heads, seq_len, seq_len)
        score = score / math.sqrt(self.d_head)
        score = tf.nn.softmax(score, axis=-1)
        score = self.dropout(score)

        output = tf.matmul(score, value)  # (batch_size, num_heads, seq_len, d_head)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.out_dense(output)


def prepare_data(x_list, y_list):
    y_list = np.array(y_list)  # 确保 y_list 是 NumPy 数组
    y_list = np.where(y_list == 5, 3, y_list)  # 将 5 映射为 3
    y = to_categorical(y_list, num_classes=4)
    return x_list, y

def balance_classes_3d(X, y, random_state=42):
    """
    平衡类别分布，确保每个类别的样本数量一致。
    :param X: 三维张量 (batch_size, timesteps, features)
    :param y: 二维 one-hot 编码标签 (batch_size, num_classes)
    :param random_state: 随机种子
    :return: 平衡后的 X 和 y
    """
    # 将 one-hot 编码转换为一维标签
    y_labels = np.argmax(y, axis=1)

    # 获取每个类别的索引
    unique_classes, counts = np.unique(y_labels, return_counts=True)
    min_samples = np.min(counts)  # 找到最小类别数量

    # 初始化平衡后的数据列表
    balanced_X = []
    balanced_y = []

    # 设置随机种子
    np.random.seed(random_state)

    # 对每个类别进行采样
    for cls in unique_classes:
        class_indices = np.where(y_labels == cls)[0]  # 获取当前类别的索引
        sampled_indices = np.random.choice(class_indices, min_samples, replace=False)  # 随机采样
        balanced_X.append(X[sampled_indices])
        balanced_y.append(y[sampled_indices])

    # 将列表合并为 NumPy 数组
    balanced_X = np.concatenate(balanced_X, axis=0)
    balanced_y = np.concatenate(balanced_y, axis=0)

    return balanced_X, balanced_y

def balance_classes(X, y, random_state=42):
    """
    平衡类别分布，确保每个类别的样本数量一致。
    :param X: 三维张量 (batch_size, timesteps, features)
    :param y: 二维 one-hot 编码标签 (batch_size, num_classes)
    :param random_state: 随机种子
    :return: 平衡后的 X 和 y
    """
    # 将 one-hot 编码转换为一维标签
    y_labels = np.argmax(y, axis=1)

    # 获取每个类别的索引
    unique_classes, counts = np.unique(y_labels, return_counts=True)
    min_samples = np.min(counts)  # 找到最小类别数量

    # 初始化平衡后的数据列表
    balanced_X = []
    balanced_y = []

    # 设置随机种子
    np.random.seed(random_state)

    # 对每个类别进行采样
    for cls in unique_classes:
        class_indices = np.where(y_labels == cls)[0]  # 获取当前类别的索引
        sampled_indices = np.random.choice(class_indices, min_samples, replace=False)  # 随机采样
        balanced_X.append(X[sampled_indices])
        balanced_y.append(y[sampled_indices])

    # 将列表合并为 NumPy 数组
    balanced_X = np.concatenate(balanced_X, axis=0)
    balanced_y = np.concatenate(balanced_y, axis=0)

    return balanced_X, balanced_y



def build_D_model(input_shape1,input_shape2, num_classes):
    # 定义 LSTM 网络的输入
    lstm_input = Input(shape=input_shape1)
    eeg_feature_input = Input(shape=input_shape2)
    # 第一层双向 LSTM
    lstm_output1 = Bidirectional(
        LSTM(units=64,
             activation='tanh',
             recurrent_activation='sigmoid',
             return_sequences=True,  # 为下一层 LSTM 保留序列维度
             kernel_regularizer=l2(1e-4)),
             name='bidirectional_lstm_1')(lstm_input)

    # 第二层双向 LSTM
    lstm_output2 = Bidirectional(
        LSTM(units=32,
             activation='tanh',
             recurrent_activation='sigmoid',
             return_sequences=False,  # 不保留序列维度
             kernel_regularizer=l2(1e-4)),
             name='bidirectional_lstm_2')(lstm_output1)

    # 全连接层
    dense3 = Dense(64, activation='swish', kernel_regularizer=l2(1e-4), name='dense_64')(lstm_output2)
    dense4 = Dense(16, activation='swish', kernel_regularizer=l2(1e-4), name='dense_16')(dense3)

    # 合并特征
    combined_features = Concatenate()([dense4, eeg_feature_input])  # 合并特征

    # 展平特征
    flattened_features = Flatten()(combined_features)
    flattened_features_dropout = Dropout(0.2)(flattened_features)

    # 添加自注意力层
    dense5 = Dense(24, activation='swish', kernel_regularizer=l2(1e-4), name='dense_24')(flattened_features_dropout)

    # 输出层
    output = Dense(num_classes, activation='softmax')(dense5)

    # 创建模型
    model = Model(inputs=[lstm_input, eeg_feature_input], outputs=output)

    # 改进点6：优化器配置
    optimizer = Adam(learning_rate=0.001,
                     beta_1=0.9,
                     beta_2=0.999,
                     clipnorm=1.0,
                     amsgrad=True)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['acc',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])

    return model




def build_D_attention_model(input_shape1,input_shape2, num_classes):
    # 定义 LSTM 网络的输入
    lstm_input = Input(shape=input_shape1)
    eeg_feature_input = Input(shape=input_shape2)
    # 第一层双向 LSTM
    lstm_output1 = Bidirectional(
        LSTM(units=64,
             activation='tanh',
             recurrent_activation='sigmoid',
             return_sequences=True,  # 为下一层 LSTM 保留序列维度
             kernel_regularizer=l2(1e-4)),
             name='bidirectional_lstm_1')(lstm_input)

    # 第二层双向 LSTM
    lstm_output2 = Bidirectional(
        LSTM(units=32,
             activation='tanh',
             recurrent_activation='sigmoid',
             return_sequences=False,  # 不保留序列维度
             kernel_regularizer=l2(1e-4)),
             name='bidirectional_lstm_2')(lstm_output1)

    # 全连接层
    dense3 = Dense(16, activation='swish', kernel_regularizer=l2(1e-4), name='dense_64')(lstm_output2)

    # 合并特征
    combined_features = Concatenate()([dense3, eeg_feature_input])  # 合并特征

    # 展平特征
    flattened_features = Flatten()(combined_features)
    flattened_features_dropout = Dropout(0.2)(flattened_features)

    # 添加自注意力层
    attention_output = SelfAttention(d_model=24)(flattened_features_dropout)

    # 如果 attention_output 是 3D 张量，使用 GlobalAveragePooling1D 消除序列维度
    if len(attention_output.shape) == 3:
        attention_output = GlobalAveragePooling1D()(attention_output)
    # 输出层
    output = Dense(num_classes, activation='softmax')(attention_output)

    # 创建模型
    model = Model(inputs=[lstm_input, eeg_feature_input], outputs=output)

    # 改进点6：优化器配置
    optimizer = Adam(learning_rate=0.001,
                     beta_1=0.9,
                     beta_2=0.999,
                     clipnorm=1.0,
                     amsgrad=True)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['acc',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])

    return model


def build_model(input_shape, num_classes):
    model = Sequential(name='BiLSTM_Classifier')

    # 改进点1：双向LSTM捕捉双向时序特征
    model.add(Bidirectional(
        LSTM(units=64,
             activation='tanh',
             recurrent_activation='sigmoid',
             return_sequences=True,  # 为下一层LSTM保留序列维度
             kernel_regularizer=l2(1e-4)),
        input_shape=input_shape
    ))

    # 改进点2：添加BatchNorm加速收敛
    model.add(BatchNormalization())

    # 改进点3：堆叠第二层LSTM提取高层特征
    model.add(Bidirectional(
        LSTM(units=32,
             activation='tanh',
             recurrent_dropout=0.2,  # 防止递归权重过拟合
             kernel_regularizer=l2(1e-4))
    ))

    # 改进点4：深度特征提取
    model.add(Dense(64, activation='swish', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.5))  # 增强泛化能力
    model.add(BatchNormalization())

    # 改进点5：自适应特征压缩
    model.add(Dense(32, activation='swish', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.3))

    # 输出层
    model.add(Dense(num_classes, activation='softmax'))

    # 改进点6：优化器配置
    optimizer = Adam(learning_rate=0.005,
                     beta_1=0.9,
                     beta_2=0.999,
                     clipnorm=1.0,
                     amsgrad=True)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


def create_sliding_window(data, labels, window_size=10):
    n_samples, n_features = data.shape
    timesteps = 2 * window_size + 1
    X = np.zeros((n_samples - 2 * window_size, timesteps, n_features))
    y = labels[window_size:n_samples - window_size]  # 裁剪标签

    for i in range(window_size, n_samples - window_size):
        start = i - window_size
        end = i + window_size + 1
        X[i - window_size] = data[start:end]

    return X, y


def lstm_attention_train(ca_df, da_df, ss_df):
    ca_x = ca_df.loc[:, eeg_channel_names].values
    da_x = da_df.loc[:, eeg_channel_names].values
    ss_x = ss_df.loc[:, eeg_channel_names].values

    ca_y = ca_df.loc[:, "Event"].values
    da_y = da_df.loc[:, "Event"].values
    ss_y = ss_df.loc[:, "Event"].values

    all_features = np.concatenate([ca_x, da_x, ss_x], axis=0)
    all_labels = np.concatenate([ca_y, da_y, ss_y], axis=0)
    X, y = prepare_data(all_features, all_labels)
    X, y = create_sliding_window(X, y, window_size=5)
    # 平衡类别分布
    X, y = balance_classes(X, y)
    # 获取输入数据的形状
    input_shape = X.shape[1:]
    model = build_model(input_shape, 4)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, restore_best_weights=True)
    # 训练模型

    history = model.fit(X, y, epochs=20, batch_size=16, verbose=1, callbacks=[early_stopping])
    plothisAccuracy(history)
    return model

def handle_all(ca_df, da_df, ss_df, loft_df):
    # 提取 EEG 数据和事件标记

    ca_eeg_df = ca_df.loc[:, eeg_channel_names]
    da_eeg_df = da_df.loc[:, eeg_channel_names]
    ss_eeg_df = ss_df.loc[:, eeg_channel_names]
    loft_eeg_df = loft_df.loc[:, eeg_channel_names]

    # 提取事件标记26
    ca_events = ca_df.loc[:, "Event"].astype(int).values.tolist()
    da_events = da_df.loc[:, "Event"].astype(int).values.tolist()
    ss_events = ss_df.loc[:, "Event"].astype(int).values.tolist()
    loft_events = loft_df.loc[:, "Event"].astype(int).values.tolist()

    # 预处理 EEG 数据
    ca_preprocessed = mne_eeg_pretreatment.handle(ca_eeg_df, ca_events)
    da_preprocessed = mne_eeg_pretreatment.handle(da_eeg_df, da_events)
    ss_preprocessed = mne_eeg_pretreatment.handle(ss_eeg_df, ss_events)
    loft_preprocessed = mne_eeg_pretreatment.handle(loft_eeg_df, loft_events)

    # 将预处理后的数据覆盖到原始 DataFrame
    ca_df.loc[:, eeg_channel_names] = ca_preprocessed.T
    da_df.loc[:, eeg_channel_names] = da_preprocessed.T
    ss_df.loc[:, eeg_channel_names] = ss_preprocessed.T
    loft_df.loc[:, eeg_channel_names] = loft_preprocessed.T

    # 返回覆盖后的 DataFrame
    return ca_df, da_df, ss_df, loft_df

def handle(ca_df, da_df, loft_df, ss_df):
    ca_eeg_df = ca_df.loc[:, eeg_channel_names]
    da_eeg_df = da_df.loc[:, eeg_channel_names]
    ss_eeg_df = ss_df.loc[:, eeg_channel_names]
    loft_eeg_df = loft_df.loc[:, eeg_channel_names]
    ca_df.loc[:, "Event"].astype(int).values.tolist()
    da_df.loc[:, "Event"].astype(int).values.tolist()
    ss_df.loc[:, "Event"].astype(int).values.tolist()
    loft_df.loc[:, "Event"].astype(int).values.tolist()
    mne_eeg_pretreatment.handle(ca_eeg_df, ca_df.loc[:, "Event"])
    mne_eeg_pretreatment.handle(da_eeg_df, da_df.loc[:, "Event"])
    mne_eeg_pretreatment.handle(ss_eeg_df, ss_df.loc[:, "Event"])
    mne_eeg_pretreatment.handle(loft_eeg_df, loft_df.loc[:, "Event"])


def plothisAccuracy(history):
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确度曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def lstm_attention(lstm_model, loft_df):
    # 提取特征和标签
    X = loft_df.loc[:, all_channel_names].values
    y_true = loft_df.loc[:, "Event"].values

    # 准备数据
    X, y_true = prepare_data(X, y_true)
    X, y_true = create_sliding_window(X, y_true, window_size=5)
    X, y_true = balance_classes(X, y_true)

    # 进行预测
    y_pred_prob = lstm_model.predict(X)  # 预测概率
    y_pred = np.argmax(y_pred_prob, axis=1)  # 将概率转换为类别标签

    accuracy,f1,report= report_handle.get_report(y_true, y_pred)


    return y_pred, accuracy, f1, report

def cebra_lstm_attention(X_train_split, cxt_embedding, y_train_split):

    model = build_D_attention_model(X_train_split.shape[1:],cxt_embedding.shape[1:], 4)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.005, patience=3, restore_best_weights=True)
    # 训练模型

    history = model.fit([X_train_split, cxt_embedding], y_train_split, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])
    plothisAccuracy(history)

    return model


def cebra_lstm(X_train_split, cxt_embedding, y_train_split):

    model = build_D_model(X_train_split.shape[1:],cxt_embedding.shape[1:], 4)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.005, patience=3, restore_best_weights=True)
    # 训练模型

    history = model.fit([X_train_split, cxt_embedding], y_train_split, epochs=100, batch_size=16, verbose=1, callbacks=[early_stopping])
    plothisAccuracy(history)

    return model


def lstm(X_train_split,y_train_split):

    # 获取输入数据的形状
    input_shape = X_train_split.shape[1:]
    model = build_model(input_shape, 4)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.005, patience=3, restore_best_weights=True)
    # 训练模型
    history = model.fit(X_train_split, y_train_split, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])
    plothisAccuracy(history)
    return model


def cebra_lstm_v3(X_train_split, y_train_split):
    # 获取输入数据的形状
    # 获取输入数据的形状
    input_shape = X_train_split.shape[1:]
    model = build_model(input_shape, 4)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.005, patience=3, restore_best_weights=True)
    # 训练模型

    history = model.fit(X_train_split, y_train_split, epochs=100, batch_size=32, verbose=1,
                        callbacks=[early_stopping])
    plothisAccuracy(history)

    return model