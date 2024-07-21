import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, recall_score

# 预测分数和真实标签
y_pred = np.array([0.0215735, 0.0490637, 0.03693893, 0.06636749, 0.02832469, 0.03551544,
                   0.0083948, 0.05424765, 0.06408712, 0.04251643, 0.02245809, 0.0120986,
                   0.01133633, 0.05839544, 0.01292803, 0.04597265, 0.03943941, 0.02300821,
                   0.07831516, 0.15442911, 0.0338998, 0.01516559, 0.01916765, 0.02718425,
                   0.02996973, 0.01733734, 0.06586169, 0.06784667, 0.01485096, 0.01713869,
                   0.01303082, 0.01304006, 0.01753993, 0.11057984, 0.16303365, 0.10694161,
                   0.09279479, 0.10086029, 0.02387015, 0.01780599, 0.03810629, 0.026802,
                   0.0260265, 0.28577614, 0.2441003, 0.21577848, 0.01059501, 0.01062025,
                   0.02728851, 0.02224658, 0.07889185, 0.06369218, 0.06304233, 0.01467722,
                   0.01649466, 0.04712306, 0.05525469, 0.03973556, 0.03560826, 0.06753475,
                   0.01497794, 0.01664076, 0.03047588, 0.01489611, 0.02191049, 0.07505499,
                   0.02138124, 0.02244142, 0.01961063, 0.017436, 0.01790555, 0.00429519,
                   0.01897446, 0.05519435, 0., 0.01796478, 1., 0.00320579,
                   0.01902794, 0.04965828])

y_true = np.array([1]*60 + [0]*20)

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_true, y_pred)

# 计算每个阈值对应的准确率和召回率
accuracies = []
recalls = []

for threshold in thresholds:
    y_pred_binary = (y_pred >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    accuracies.append(acc)
    recalls.append(recall)

# 找到最佳阈值
best_threshold_index = np.argmax(accuracies)
best_threshold = thresholds[best_threshold_index]

print(f'Best Threshold: {best_threshold}')
print(f'Accuracy at Best Threshold: {accuracies[best_threshold_index]}')
print(f'Recall at Best Threshold: {recalls[best_threshold_index]}')
