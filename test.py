import streamlit as st
import pandas as pd
from sklearn import preprocessing
import numpy as np
from pycaret.classification import *
Gender = 1
Smoking_history = 1
Hypertension = 1
Hyperlipemia = 1
Coronary_heart_disease = 1
Auricular_fibrillation = 1
Diabetes = 1
Cerebral_stroke = 1
Hypertensive_heart_disease = 1

Age = 50
SBP = 120
DBP = 80
HB = 13.5
AST = 25
ALB = 4.0
Urea = 7.0
Creatinine = 1.1
Cl = 102
GLU = 20
FS = 5.0
LAD = 35
RATD = 20
RAD = 35
category_data=[Gender, Smoking_history, Hypertension,
    Hyperlipemia, Coronary_heart_disease, Auricular_fibrillation, Diabetes,
    Cerebral_stroke, Hypertensive_heart_disease]

numerical_data=[Age, SBP, DBP, HB, AST, ALB, Urea, Creatinine, Cl, GLU,FS, LAD, RATD, RAD]

numerical_data = np.array(numerical_data).reshape(1, -1)  # 转换为二维数组

min_max_scaler = preprocessing.MinMaxScaler()
numerical_data = min_max_scaler.fit_transform(numerical_data)

format_data = numerical_data.flatten().tolist() + category_data


RP1_model = load_model(model_name='PhenoGroup1-RP')
feature_names = ['Age', 'SBP', 'DBP', 'HB', 'AST', 'ALB', 'Urea', 'Creatinine', 'Cl', 'GLU', 'FS', 'LAD', 'RATD', 'RAD','Gender', 'Smoking_history', 'Hypertension', 'Hyperlipemia', 'Coronary_heart_disease',
                 'Auricular_fibrillation', 'Diabetes', 'Cerebral_stroke', 'Hypertensive_heart_disease']
format_data_df = pd.DataFrame(data=[format_data], columns=feature_names)
# 使用模型对格式化后的数据format_data进行预测,返回预测的类别代码
prediction = RP1_model.predict(format_data_df)
prediction_proba = RP1_model.predict_proba(format_data_df)
print(type(prediction_proba))
print("Shape of prediction:", prediction_proba.shape)
print(prediction)
print( prediction_proba)




