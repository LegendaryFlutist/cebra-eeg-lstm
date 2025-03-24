import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline


def rename_channels(raw):
    # 定义名称映射
    rename_dict = {
        'EEG_FP1': 'Fp1',
        'EEG_F7': 'F7',
        'EEG_F8': 'F8',
        'EEG_T4': 'T8',
        'EEG_T6': 'P8',
        'EEG_T5': 'P7',
        'EEG_T3': 'T7',
        'EEG_FP2': 'Fp2',
        'EEG_O1': 'O1',
        'EEG_P3': 'P3',
        'EEG_Pz': 'Pz',
        'EEG_F3': 'F3',
        'EEG_Fz': 'Fz',
        'EEG_F4': 'F4',
        'EEG_C4': 'C4',
        'EEG_P4': 'P4',
        'EEG_POz': 'POz',
        'EEG_C3': 'C3',
        'EEG_Cz': 'Cz',
        'EEG_O2': 'O2'
    }

    # 重命名电极
    raw.rename_channels(rename_dict)


def csp(epochs):
    epochs_data = epochs.get_data()
    epochs_train = epochs.copy().crop(tmin=-0.2, tmax=0.5)
    labels = epochs.events[:, -1] - 2
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_validate function
    clf = Pipeline([("CSP", csp), ("LDA", lda)])

    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    # Perform cross-validation
    scores = cross_validate(clf, epochs_data_train, labels, cv=cv, scoring=scoring, n_jobs=None)

    # Printing the results
    print("准确率:", np.mean(scores['test_accuracy']))
    print("精确率:", np.mean(scores['test_precision']))
    print("召回率:", np.mean(scores['test_recall']))
    print("F1 分数:", np.mean(scores['test_f1']))
    print("AUC:", np.mean(scores['test_roc_auc']))

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)
    csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

def handle(ca_eeg_df, events):
    events_markers = events
    event_samples = np.arange(len(events_markers))
    event_durations = np.zeros(len(events_markers))
    # events = np.column_stack((event_samples, event_durations, events_markers))
    # events = events.astype(int)
    sfreq = 256  # 假设采样频率是256Hz，根据实际情况修改
    ch_names = ca_eeg_df.columns.tolist()  # 获取通道名
    info = mne.create_info(ch_names=ch_names, ch_types=['eeg'] * 20, sfreq=sfreq)

    # 将DataFrame转换为MNE的RawArray对象
    data = ca_eeg_df.values
    raw = mne.io.RawArray(np.transpose(data), info)

    # 带通滤波，例如设置频率范围为1-40Hz，可根据实际需求修改
    l_freq = 1
    h_freq = 40
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
    rename_channels(raw)

    # 去除眼电伪迹（使用 ICA）
    ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter=800)
    ica.fit(raw)
    ica.apply(raw)

    # 电极信息
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

    #plt_row(raw)

    # 4. 重参考（设置为平均参考）
    raw.set_eeg_reference('average', projection=True)


    # 返回预处理后的数据
    return raw.get_data()



def plt_row(raw):
    # 生成的原始数据波形图
    raw.plot(start=1, duration=1, n_channels=32, clipping=None)
    plt.show()
    # 原始数据功率谱图
    raw.plot_psd(fmax=50, average=True)
    plt.show()
    # 电极位置图
    raw.plot_sensors(ch_type='eeg', show_names=True)
    plt.show()
    # 3. 计算功率谱密度（PSD）
    psds, freqs = raw.compute_psd(fmax=40).get_data(return_freqs=True)

    # 4. 定义频段
    bands = {
        'Delta (1-4 Hz)': (1, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-40 Hz)': (30, 40)
    }

    # 5. 创建地形图布局
    fig, axes = plt.subplots(1, len(bands), figsize=(20, 4))
    fig.suptitle('Power Spectral Topography', fontsize=16)

    # 6. 绘制每个频段的地形图
    for ax, (title, (fmin, fmax)) in zip(axes, bands.items()):
        # 找到对应频段的索引
        freq_mask = (freqs >= fmin) & (freqs <= fmax)

        # 计算频段内平均功率
        band_power = psds[:, freq_mask].mean(axis=1)

        # 绘制地形图
        mne.viz.plot_topomap(
            band_power,
            raw.info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            vlim=(np.percentile(band_power, 5), np.percentile(band_power, 95))  # 使用 vlim 替代 limits
        )
        ax.set_title(title)

    # 添加颜色条
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    cbar = fig.colorbar(ax.images[0], cax=cax, orientation='horizontal')
    cbar.set_label('Power Spectral Density (dB)')

    plt.show()

def plot_epochs(epochs):
    # 绘制 epoch 数据
    epochs.plot_image(picks='eeg', combine='mean', title='Epochs Image Plot')
    plt.show()

def plot_evoked(epochs):
    # 计算 evoked 数据
    evoked = epochs.average()
    # 绘制 evoked 数据
    evoked.plot(spatial_colors=True, titles='Evoked Response')
    plt.show()
    # 绘制地形图
    evoked.plot_topomap(times=[0.1, 0.2, 0.3, 0.4])
    plt.show()

def plot_time_frequency(epochs):
    # 计算时频表示
    freqs = np.arange(2, 40, 2)  # 频率范围
    n_cycles = freqs / 2.  # 每个频率的周期数
    power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)
    # 绘制时频图