import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler
# 读取图像
image = cv2.imread(r'E:\python\project\AI_course-master\img.jpg')
# 提取特征向量
features = image.reshape(-1, 3)  # 将图像展平为一维, 3通道向量
# 标注数据
# 这里假设已经有了感兴趣区域的掩码图像，即二进制图像，其中像素值为1表示感兴趣区域，像素值为0(黑色)表示背景
mask = cv2.imread(r'E:\python\project\AI_course-master\output.png', 0)
labels = mask.flatten()  # 将掩码图像展平为一维标签
# 进行随机欠采样处理特征向量
rus = RandomUnderSampler(random_state=42)
features_resampled, labels_resampled = rus.fit_resample(features, labels)
# 数据划分
X_train, X_test, y_train, y_test = train_test_split(features_resampled, labels_resampled, test_size=0.2, random_state=42)
# 创建逻辑回归模型
logreg = LogisticRegression()
# 模型训练
logreg.fit(X_train, y_train)
# 模型预测
y_pred_train = logreg.predict(X_train)
y_pred_test = logreg.predict(X_test)
# 模型评估
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# print("Train Accuracy:", accuracy_train)
# print("Test Accuracy:", accuracy_test)
# accuracy_train = accuracy_score(y_train, y_pred_train)
# accuracy_test = accuracy_score(y_test, y_pred_test)
# _auc=roc_auc_score(y_test, y_pred_test)
# precision = precision_score(y_test, y_pred_test,pos_label=38)
# recall = recall_score(y_test, y_pred_test,pos_label=38)
# f1 = f1_score(y_test, y_pred_test,pos_label=38)
# # print("Train Accuracy:", accuracy_train)
# # print("Test Accuracy:", accuracy_test)
# print('auc:',_auc)
# print('查准率',precision)
# print('查全率',recall)
# print('F1',f1)
# 应用模型进行图像分割
predicted_labels = logreg.predict(features)
segmentation_mask = predicted_labels.reshape(image.shape[:2])
cv2.imshow("Image segmentation",segmentation_mask)
cv2.waitKey(0)

test_image = cv2.imread(r'E:\python\project\AI_course-master\img1.jpg')
test_image2 = cv2.imread(r'E:\python\project\AI_course-master\img1.jpg')
test_image = test_image.reshape(-1,3)
out = logreg.predict(test_image).reshape(test_image2.shape[:2])
cv2.imwrite("Image_segmentation.jpg",out)