import numpy as np
from keras.src.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassMetricsCallback(Callback):
    def __init__(self, X_val, y_val):
        super(ClassMetricsCallback, self).__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val)
        y_pred_classes = np.argmax(y_pred, axis=-1)
        y_true_classes = np.argmax(self.y_val, axis=-1)

        # 计算每个类别的指标
        for i in range(4):  # 4 个类别
            class_mask = (y_true_classes == i)
            if np.sum(class_mask) > 0:  # 确保类别存在
                acc = accuracy_score(y_true_classes[class_mask], y_pred_classes[class_mask])
                precision = precision_score(y_true_classes[class_mask], y_pred_classes[class_mask], average='binary')
                recall = recall_score(y_true_classes[class_mask], y_pred_classes[class_mask], average='binary')
                f1 = f1_score(y_true_classes[class_mask], y_pred_classes[class_mask], average='binary')
                print(f"类别 {i} - 准确率: {acc:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1 分数: {f1:.4f}")
            else:
                print(f"类别 {i} - 无样本")
