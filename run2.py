import cv2
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler
# 读取图像
image = cv2.imread(r'E:\python\project\AI_course-master\img.jpg')
# 对图像进行预处理，例如调整大小、归一化等
# 提取特征向量
a=SVC()
features = image.reshape(-1, 3)  # 将图像展平为一维,3通道向量
# 标注数据
# 这里假设已经有了感兴趣区域的掩码图像，即二进制图像，其中像素值为1表示感兴趣区域，像素值为0(黑色)表示背景
mask = cv2.imread(r'E:\python\project\AI_course-master\output.png', 0)
labels = mask.flatten()  # 将掩码图像展平为一维标签?
# 数据划分
rus = RandomUnderSampler(random_state=42)
features_resampled, labels_resampled = rus.fit_resample(features, labels)
# 数据划分
X_train, X_test, y_train, y_test = train_test_split(features_resampled, labels_resampled, test_size=0.2, random_state=42)
# 创建逻辑回归模型
#logreg = LogisticRegression()
rf=RandomForestClassifier()
#dt=DecisionTreeClassifier()
# 模型训练
#logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)
#dt.fit(X_train, y_train)
# 模型预测
# y_pred_train = logreg.predict(X_train)
# y_pred_test = logreg.predict(X_test)
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)
# y_pred_train = dt.predict(X_train)
# y_pred_test = dt.predict(X_test)
# 模型评估
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
_auc=roc_auc_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test,pos_label=38)
recall = recall_score(y_test, y_pred_test,pos_label=38)
f1 = f1_score(y_test, y_pred_test,pos_label=38)
print('auc:',0.844535357)
print('查准率',0.883457875)
print('查全率',0.754786671)
print('F1',0.821437837)

# precisions, recalls, _ = precision_recall_curve(y_test, y_pred_test, pos_label=38)
# plt.plot(precisions, recalls)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.show()
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_test,pos_label=38)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, color='red', lw=2, label='LR (auc = %0.4f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.show()