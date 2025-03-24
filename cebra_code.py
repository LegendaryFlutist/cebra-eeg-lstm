import os
import torch
from cebra import CEBRA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP
# 新增标准化与滑动窗口处理

from sklearn.preprocessing import RobustScaler, StandardScaler
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
plotDirDev = "." + os.sep + "plot"

def plt3D(ca_embedding, da_embedding, ss_embedding, non_embedding, type, text, p_num):
    # 可视化优化（基于Nature期刊配色标准）[2]()
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # CA（深蓝）: 科研常用主色
    ax.scatter(ca_embedding[:, 0], ca_embedding[:, 1], ca_embedding[:, 2],
               label='CA', c='#2E5D87', alpha=0.7, s=18, edgecolor='w', linewidth=0.3)

    # DA（橙红）: 高对比度辅助色
    ax.scatter(da_embedding[:, 0], da_embedding[:, 1], da_embedding[:, 2],
               label='DA', c='#D1495B', alpha=0.7, s=18, marker='s', edgecolor='w', linewidth=0.3)

    # SS（青绿）: 保留原色系但调整饱和度
    ax.scatter(ss_embedding[:, 0], ss_embedding[:, 1], ss_embedding[:, 2],
               label='SS', c='#4A9C7D', alpha=0.7, s=18, marker='^', edgecolor='w', linewidth=0.3)

    # NONE（中性灰）: 非事件类标准化处理
    ax.scatter(non_embedding[:, 0], non_embedding[:, 1], non_embedding[:, 2],
               label='NONE', c='#7F7F7F', alpha=0.5, s=12, marker='x', linewidth=0.8)

    # 坐标轴与图例优化
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    ax.grid(linestyle=':', color='#666666', alpha=0.4)

    # 专业级图例排版
    legend = ax.legend(
        title="Event Types",
        title_fontsize=12,
        fontsize=10,
        markerscale=1.8,
        frameon=True,
        framealpha=0.9,
        edgecolor='#333333',
        loc='upper right'
    )
    legend.get_title().set_fontweight('bold')

    # 添加色盲友好标识[6]()
    ax.text2D(0.02, 0.95, "✓ CVD-safe Colors",
              transform=ax.transAxes,
              color='#444444',
              fontsize=9,
              alpha=0.8)

    plt.title('CEBRA_'+text+' 3D Event Embedding (Optimized Color Scheme)',
              fontsize=14,
              pad=20)
    plt.tight_layout()

    # 调整保存顺序（先保存后显示）
    fig.savefig(
        plotDirDev + os.sep + "cebra_" +text + "_" + type + "_" + p_num + ".png",
        dpi=300,  # 印刷级分辨率
        bbox_inches='tight',  # 去除多余白边
        facecolor=fig.get_facecolor(),  # 继承figure背景色
        transparent=False,  # 关闭透明通道
        pil_kwargs={'optimize': True}  # PNG优化
    )
    # 显示验证
    plt.show()


def plot_cebra(result_dfs, p_num, cebra_model,n = 500, type = 'baseline'):
    loft_non = result_dfs["event_0_df"].sample(n=n, random_state=42, replace=False).loc[:, eeg_channel_names].to_numpy()
    loft_ss = result_dfs["event_1_df"].sample(n=n, random_state=42, replace=False).loc[:, eeg_channel_names].to_numpy()
    loft_ca = result_dfs["event_2_df"].sample(n=n, random_state=42, replace=False).loc[:, eeg_channel_names].to_numpy()
    loft_da = result_dfs["event_5_df"].sample(n=n, random_state=42, replace=False).loc[:, eeg_channel_names].to_numpy()

    ca_embedding = cebra_model.transform(loft_ca)
    da_embedding = cebra_model.transform(loft_da)
    ss_embedding = cebra_model.transform(loft_ss)
    non_embedding = cebra_model.transform(loft_non)
    # 数据标准化
    scaler = StandardScaler()
    ca_embedding_scaled = scaler.fit_transform(ca_embedding)
    da_embedding_scaled = scaler.fit_transform(da_embedding)
    ss_embedding_scaled = scaler.fit_transform(ss_embedding)
    non_embedding_scaled = scaler.fit_transform(non_embedding)

    # 使用t-SNE进行降维
    # tsne = TSNE(n_components=3, random_state=42)
    # ca_embeddings_tsne = tsne.fit_transform(ca_embedding_scaled)
    # da_embeddings_tsne = tsne.fit_transform(da_embedding_scaled)
    # ss_embeddings_tsne = tsne.fit_transform(ss_embedding_scaled)
    # non_embeddings_tsne = tsne.fit_transform(non_embedding_scaled)

    # 使用UMAP进行降维
    umap_model = UMAP(n_components=3, random_state=42)
    ca_embeddings_umap = umap_model.fit_transform(ca_embedding_scaled)
    da_embeddings_umap = umap_model.fit_transform(da_embedding_scaled)
    ss_embeddings_umap = umap_model.fit_transform(ss_embedding_scaled)
    non_embeddings_umap = umap_model.fit_transform(non_embedding_scaled)

    # plt3D(ca_embedding, da_embedding, ss_embedding, non_embedding, type, "", p_num)
    # plt3D(ca_embeddings_tsne, da_embeddings_tsne, ss_embeddings_tsne, non_embeddings_tsne, type, "t-sne", p_num)
    plt3D(ca_embeddings_umap, da_embeddings_umap, ss_embeddings_umap, non_embeddings_umap, type, "umap", p_num)
    pass


def cebra_model(result_dfs):
    n = max(len(result_dfs["event_1_df"]), len(result_dfs["event_2_df"]), len(result_dfs["event_5_df"]))
    non = result_dfs["event_0_df"].reset_index(drop=True).loc[:n, eeg_channel_names].to_numpy()
    ss = result_dfs["event_1_df"].loc[:, eeg_channel_names].to_numpy()
    ca = result_dfs["event_2_df"].loc[:, eeg_channel_names].to_numpy()
    da = result_dfs["event_5_df"].loc[:, eeg_channel_names].to_numpy()

    # 合并数据并生成连续时间索引（关键步骤！）
    train_data = np.concatenate((non, ca, da, ss), axis=0)
    time_index = np.arange(len(train_data))  # 生成全局连续时间索引

    # 创建模型 (关键参数优化)[5]()
    # 修改后的模型配置（启用监督模式） 时间对比学习
    cebra_model = CEBRA(
        model_architecture='offset10-model',
        batch_size=256,
        learning_rate=1e-3,
        temperature_mode='auto',
        output_dimension=12,  # 先高维再降维
        max_iterations=15000,
        conditional='time_delta',  # 多模态对比
        distance='cosine',  # 更适合时序相似性计算
        hybrid= True,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True,
        time_offsets=10
    )

    # 合并训练数据 (CA/DA/SS)
    train_data = np.concatenate((non, ca, da, ss), axis=0)
    behavior_labels = np.concatenate([
        np.zeros(len(non)),  # non 对应标签 0
        np.full(len(ca), 2),  # ca 对应标签 2
        np.full(len(da), 3),  # ss 对应标签 3
        np.ones(len(ss))  # ss 对应标签 1
    ], axis=0)
    # 训练与转换
    # 合并行为标签（假设labels为与train_data对应的类别）
    multi_label = np.concatenate([time_index[:, None], behavior_labels[:, None]], axis=1)

    torch.cuda.empty_cache()  # 清理未使用的缓存
    cebra_model.fit(train_data, multi_label)

    return cebra_model,train_data,behavior_labels


def init_data(ca_df, da_df, loft_df, ss_df):
    merged_df = pd.concat([ca_df, da_df, ss_df], axis=0)
    merged_df['Event'] = merged_df['Event'].astype(int)
    loft_df['Event'] = loft_df['Event'].astype(int)
    target_events = [0, 1, 2, 5]
    result_dfs = {}
    result_loft_dfs = {}
    for event in target_events:
        # 筛选目标事件数据
        event_data = merged_df[merged_df['Event'] == event]
        result_dfs[f"event_{event}_df"] = event_data
        event_loft_df = loft_df[loft_df['Event'] == event]
        result_loft_dfs[f"event_{event}_df"] = event_loft_df
    return result_dfs, result_loft_dfs


from scipy.ndimage import gaussian_filter


def create_fluid_density(embeddings, grid_size=50, sigma=2.0):
    """将离散点云转化为连续密度场"""
    # 核密度估计
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel='gaussian', bandwidth=0.3)
    kde.fit(embeddings)

    # 生成三维网格
    x = np.linspace(embeddings[:, 0].min(), embeddings[:, 0].max(), grid_size)
    y = np.linspace(embeddings[:, 1].min(), embeddings[:, 1].max(), grid_size)
    z = np.linspace(embeddings[:, 2].min(), embeddings[:, 2].max(), grid_size)
    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # 计算概率密度并平滑
    density = np.exp(kde.score_samples(grid_points))
    density = density.reshape(xx.shape)
    return gaussian_filter(density, sigma=sigma), (x, y, z)