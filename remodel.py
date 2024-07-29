import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
from sklearn.svm import SVC
import numpy as np
import pandas as pd
cluster_df = pd.read_excel("表型识别器所用数据.xlsx")
cluster_df = cluster_df.iloc[:, 2:]
np.random.seed(715)
cluster_label_df = cluster_df['Cluster'] #以cuisine列为预测标签
cluster_feature_df = cluster_df.drop([ 'Cluster'], axis=1) #删去无用的列，取食材列为可训练特征
X_train, X_test, Y_train, Y_test = train_test_split(cluster_feature_df, cluster_label_df, test_size=0.3)#以7：3比例划分训练集与测试集
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
import numpy as np

C = 10  # 正则化参数设置为10
# Create different classifiers.
classifiers = {
    'Decision Tree Classifier': DecisionTreeClassifier(random_state=715),
    # 假设您的特征是数值型的，则使用GaussianNB
    # 如果您的特征是离散的（例如，文本分类），则可能想要使用MultinomialNB
    'Gaussian Naive Bayes': GaussianNB(),
    # 'Multinomial Naive Bayes': MultinomialNB(),
    'Neural Network Classifier': MLPClassifier(random_state=715),
    'KNN classifier': KNeighborsClassifier(n_neighbors=3),
    'SVC': SVC(random_state=2),
    'RFST': RandomForestClassifier(n_estimators=100, random_state=715),

}  # 创建分类器列表
n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, np.ravel(Y_train))

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)

    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(Y_test, y_pred))

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(X_train, np.ravel(Y_train))
import pickle

# 假设你已经训练了随机森林模型并且存储在变量rf_model中
# rf_model = RandomForestClassifier(n_estimators=100, random_state=715)
# rf_model.fit(X_train, np.ravel(Y_train))

# 保存模型到文件
with open('HFpEF phenogroup Identifier2.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)

print("Random Forest model saved successfully.")