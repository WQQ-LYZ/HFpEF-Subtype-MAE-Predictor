# 第八章/streamlit_predict_v2.py
import streamlit as st
import pickle
import pandas as pd
from sklearn import preprocessing
import numpy as np
from pycaret.classification import *

# 设置页面的标题、图标和布局
st.set_page_config(
    page_title="HFpEF PhenoRiskAssist——Precision Medicine Starts Here",  # 页面标题
    page_icon="heart_icon.png",  # 页面图标
    layout='wide',
)
# 使用侧边栏实现多页面效果
with st.sidebar:
    st.image('right_logo.png', width=100)
    st.title('Please select the page')
    page = st.selectbox("Please select the page", ["Introduction", "HFpEF PhenoRiskAssist"], label_visibility='collapsed')

if page == "Introduction":
    st.title("Welcome to HFpEF PhenoRiskAssist！")

    import streamlit as st

    st.markdown("""<p style="font-size:18px; line-height:2em;">
        <strong>HFpEF PhenoRiskAssist</strong> is an innovative clinical decision support tool designed to identify phenotypes and 
        predict prognosis for HFpEF patients through unsupervised machine learning algorithms. This platform is built on real-world data, providing clinicians with precise patient care and management.
    </p>""", unsafe_allow_html=True)
    st.header('Five Phenogroups')
    st.image('Phenogroup.jpg')
    import streamlit as st

    import streamlit as st

    st.markdown("""<p style="font-size:18px; line-height:2em;">
        <strong>Disclaimer</strong>
        THE AUTHORS, OWNERS, AND PROVIDERS OF THE HFpEF PHENORISKASSIST TOOL DISCLAIM ANY LIABILITY TO ANY PATIENT OR ANY OTHER PERSON. BY USING THIS TOOL, YOU AGREE THAT YOU HAVE READ, UNDERSTOOD, AND ACCEPT THE TERMS OF THIS DISCLAIMER.
        <br>
        THE INFORMATION AND PREDICTIONS PROVIDED BY HFpEF PHENORISKASSIST ARE FOR EDUCATIONAL AND INFORMATIONAL PURPOSES ONLY AND ARE NOT INTENDED TO REPLACE PROFESSIONAL MEDICAL ADVICE. THE TOOL IS OFFERED "AS IS" WITHOUT ANY WARRANTY, EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
        <br>
        ALL LIABILITY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES ARISING FROM THE USE OF THIS TOOL IS HEREBY DISCLAIMED AND EXCLUDED. THE AUTHORS, OWNERS, AND PROVIDERS ASSUME NO RESPONSIBILITY FOR ANY CONSEQUENCE RESULTING FROM THE USE OF THIS TOOL.
    </p>""", unsafe_allow_html=True)
    st.markdown("""<p style='font-size:18px; line-height:2em;'>
        HFpEF PhenoRiskAssist is designed to supplement, not replace, the professional judgment of healthcare providers. The tool considers a variety of factors but may not account for all potentially difficult-to-measure variables that could impact an individual patient's risk of an adverse outcome. Patients deemed to be at low risk may still experience major adverse events, and there is no subgroup with a zero risk of complications.
        This tool is made available free of charge for use by physicians in providing individual patient care. Any commercial use of the tool without permission is strictly prohibited. For licensing inquiries, please contact Chongqing Medical University at <a href='mailto:fernsmith521@gmail.com'>fernsmith521@gmail.com</a>.
    </p>""", unsafe_allow_html=True)
elif page == "HFpEF PhenoRiskAssist":
   # 该页面是3:1:2的列布局
    col_form, col = st.columns([4, 1])
    predict_result_code_FULL=0
    with (col_form):
        st.header("HFpEF phenogroup Identifier")
        st.markdown(
            "This web app uses data from Chongqing Medical University to predict HFpEF phenotypes."
            " Input 23 details to begin.")

        # 运用表单和表单提交按钮
        with st.form('user_inputs'):
            Gender = st.selectbox('Gender', options=['Male', 'Female'])
            Smoking_history = st.selectbox('Smoking History', options=['Yes', 'No'])
            Hypertension = st.selectbox('Hypertension', options=['Yes', 'No'])
            Hyperlipemia = st.selectbox('Hyperlipemia', options=['Yes', 'No'])
            Coronary_heart_disease = st.selectbox('Coronary Heart Disease', options=['Yes', 'No'])
            Auricular_fibrillation = st.selectbox('Auricular Fibrillation', options=['Yes', 'No'])
            Diabetes = st.selectbox('Diabetes', options=['Yes', 'No'])
            Cerebral_stroke = st.selectbox('Cerebral Stroke', options=['Yes', 'No'])
            Hypertensive_heart_disease = st.selectbox('Hypertensive Heart Disease', options=['Yes', 'No'])

            Age = st.number_input('Age,years', min_value=18)
            SBP = st.number_input('SBP,mmHg', min_value=0.0)
            DBP = st.number_input('DBP,mmHg', min_value=0.0)
            HB = st.number_input('HB,g/dL', min_value=0.0)
            AST = st.number_input('AST,U/L', min_value=0.0)
            ALB = st.number_input('ALB,g/dL', min_value=0.0)
            Urea = st.number_input('Urea,mmol/L', min_value=0.0)
            Creatinine = st.number_input('Creatinine,mmol/L', min_value=0.0)
            Cl = st.number_input('Cl,mmol/L', min_value=0.0)
            GLU = st.number_input('GLU,mmol/L', min_value=0.0)
            FS = st.number_input('FS,%', min_value=0.0)
            LAD = st.number_input('LAD,mm)', min_value=0.0)
            RATD = st.number_input('RATD,mm)', min_value=0.0)
            RAD = st.number_input('RAD,mm)', min_value=0.0)

            submitted = st.form_submit_button('predict')

    # 初始化数据预处理格式中岛屿相关的变量
    if Gender == 'Male':
        Gender = 1
    elif Gender == 'Female':
        Gender = 0

    if Smoking_history == 'Yes':
        Smoking_history = 1
    elif Smoking_history == 'No':
        Smoking_history = 0

    if Hypertension == 'Yes':
        Hypertension = 1
    elif Hypertension == 'No':
        Hypertension = 0

    if Hyperlipemia == 'Yes':
        Hyperlipemia = 1
    elif Hyperlipemia == 'No':
        Hyperlipemia = 0

    if Coronary_heart_disease == 'Yes':
        Coronary_heart_disease = 1
    elif Coronary_heart_disease == 'No':
        Coronary_heart_disease = 0

    if Auricular_fibrillation == 'Yes':
        Auricular_fibrillation = 1
    elif Auricular_fibrillation == 'No':
        Auricular_fibrillation = 0

    if Diabetes == 'Yes':
        Diabetes = 1
    elif Diabetes == 'No':
        Diabetes = 0

    if Cerebral_stroke == 'Yes':
        Cerebral_stroke = 1
    elif Cerebral_stroke == 'No':
        Cerebral_stroke = 0

    if Hypertensive_heart_disease == 'Yes':
        Hypertensive_heart_disease = 1
    elif Hypertensive_heart_disease == 'No':
        Hypertensive_heart_disease = 0

    category_data=[Gender, Smoking_history, Hypertension,
    Hyperlipemia, Coronary_heart_disease, Auricular_fibrillation, Diabetes,
    Cerebral_stroke, Hypertensive_heart_disease]

    numerical_data=[Age, SBP, DBP, HB, AST, ALB, Urea, Creatinine, Cl, GLU,FS, LAD, RATD, RAD]

    numerical_data = np.array(numerical_data).reshape(1, -1)  # 转换为二维数组

    min_max_scaler = preprocessing.MinMaxScaler()
    numerical_data = min_max_scaler.fit_transform(numerical_data)

    format_data =   numerical_data.flatten().tolist() + category_data




    # 使用pickle的load方法从磁盘文件反序列化加载一个之前保存的随机森林模型对象

    with open('HFpEF phenogroup Identifier2.pkl', 'rb') as f:
        rfc_model = pickle.load(f)
    # 使用pickle的load方法从磁盘文件反序列化加载一个之前保存的映射对象
    # with open('output_uniques.pkl', 'rb') as f:
    #     output_uniques_map = pickle.load(f)



    if submitted:
        format_data_df = pd.DataFrame(data=[format_data], columns=rfc_model.feature_names_in_)
        # 使用模型对格式化后的数据format_data进行预测,返回预测的类别代码
        predict_result_code = rfc_model.predict(format_data_df)

        # 根据预测结果输出对应的企鹅物种名称
        if submitted:
            format_data_df = pd.DataFrame(data=[format_data], columns=rfc_model.feature_names_in_)
            # 使用模型对格式化后的数据format_data进行预测,返回预测的类别代码
            predict_result_code = rfc_model.predict(format_data_df)

            # 根据预测结果输出对应的企鹅物种名称
            if predict_result_code == 1:
                predict_result_code_FULL = 1
                st.write(f'Predict the phenotype of this HFpEF patient：**Male**\n**Low Risk**\n**Phenogroup**')
            elif predict_result_code == 2:
                predict_result_code_FULL = 2
                st.write(f'Predict the phenotype of this HFpEF patient：**Male**\n**Atherosclerosis**\n**Phenogroup**')
            elif predict_result_code == 3:
                predict_result_code_FULL = 3
                st.write(f'Predict the phenotype of this HFpEF patient：**Female**\n**Diabetes**\n**Phenogroup**')
            elif predict_result_code == 4:
                predict_result_code_FULL = 4
                st.write(f'Predict the phenotype of this HFpEF patient：**Female**\n**Atrial Fibrillation**\n**Phenogroup**')
            elif predict_result_code == 5:
                predict_result_code_FULL = 5
                st.write(f'Predict the phenotype of this HFpEF patient：**Female**\n**Low Risk**\n**Phenogroup**')
            else:
                st.write(f'Prediction outcome unknown based on input data.')

        st.header("HFpEF phenogroup Risk Predictors")
        if predict_result_code_FULL == 1:
            RP1_model = load_model(model_name='PhenoGroup1-RP')
            feature_names = ['Age', 'SBP', 'DBP', 'HB', 'AST', 'ALB', 'Urea', 'Creatinine', 'Cl','GLU','FS', 'LAD', 'RATD','RAD','Gender', 'Smoking_history', 'Hypertension', 'Hyperlipemia', 'Coronary_heart_disease', 'Auricular_fibrillation', 'Diabetes', 'Cerebral_stroke', 'Hypertensive_heart_disease']
            format_data_df = pd.DataFrame(data=[format_data], columns=feature_names)
            prediction = RP1_model.predict(format_data_df)
            prediction_proba = RP1_model.predict_proba(format_data_df)
            if prediction == 1:
                st.markdown(
                    f'**The patient will have an adverse prognosis, with a probability of:** <span style="font-size:24px; font-weight:bold;">{1-prediction_proba[0][0]:.2f}</span>',
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    f'**The patient will not have an adverse prognosis, with a probability of:** <span style="font-size:24px; font-weight:bold;">{prediction_proba[0][0]:.2f}</span>',
                    unsafe_allow_html=True
                )
        elif predict_result_code_FULL == 2:
            RP2_model = load_model(model_name='PhenoGroup2-RP')
            feature_names = ['Age', 'SBP', 'DBP', 'HB', 'AST', 'ALB', 'Urea', 'Creatinine', 'Cl','GLU','FS', 'LAD', 'RATD','RAD','Gender', 'Smoking_history', 'Hypertension', 'Hyperlipemia', 'Coronary_heart_disease', 'Auricular_fibrillation', 'Diabetes', 'Cerebral_stroke', 'Hypertensive_heart_disease']
            format_data_df = pd.DataFrame(data=[format_data], columns=feature_names)
            prediction = RP2_model.predict(format_data_df)
            prediction_proba = RP2_model.predict_proba(format_data_df)
            if prediction == 1:
                st.markdown(
                    f'**The patient will have an adverse prognosis, with a probability of:** <span style="font-size:24px; font-weight:bold;">{1-prediction_proba[0][0]:.2f}</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'**The patient will not have an adverse prognosis, with a probability of:** <span style="font-size:24px; font-weight:bold;">{prediction_proba[0][0]:.2f}</span>',
                    unsafe_allow_html=True
                )
        elif predict_result_code_FULL == 3:
            RP3_model = load_model(model_name='PhenoGroup3-RP')
            feature_names = ['Age', 'SBP', 'DBP', 'HB', 'AST', 'ALB', 'Urea', 'Creatinine', 'Cl','GLU','FS', 'LAD', 'RATD','RAD','Gender', 'Smoking_history', 'Hypertension', 'Hyperlipemia', 'Coronary_heart_disease', 'Auricular_fibrillation', 'Diabetes', 'Cerebral_stroke', 'Hypertensive_heart_disease']
            format_data_df = pd.DataFrame(data=[format_data], columns=feature_names)
            prediction = RP3_model.predict(format_data_df)
            prediction_proba = RP3_model.predict_proba(format_data_df)
            if prediction == 1:
                st.markdown(
                    f'**The patient will have an adverse prognosis, with a probability of:** <span style="font-size:24px; font-weight:bold;">{1-prediction_proba[0][0]:.2f}</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'**The patient will not have an adverse prognosis, with a probability of:** <span style="font-size:24px; font-weight:bold;">{prediction_proba[0][0]:.2f}</span>',
                    unsafe_allow_html=True
                )
        elif predict_result_code_FULL == 4:
            RP4_model = load_model(model_name='PhenoGroup4-RP')
            feature_names = ['Age', 'SBP', 'DBP', 'HB', 'AST', 'ALB', 'Urea', 'Creatinine', 'Cl','GLU','FS', 'LAD', 'RATD','RAD','Gender', 'Smoking_history', 'Hypertension', 'Hyperlipemia', 'Coronary_heart_disease', 'Auricular_fibrillation', 'Diabetes', 'Cerebral_stroke', 'Hypertensive_heart_disease']
            format_data_df = pd.DataFrame(data=[format_data], columns=feature_names)
            prediction = RP4_model.predict(format_data_df)
            prediction_proba = RP4_model.predict_proba(format_data_df)
            if prediction == 1:
                st.markdown(
                    f'**The patient will have an adverse prognosis, with a probability of:** <span style="font-size:24px; font-weight:bold;">{1-prediction_proba[0][0]:.2f}</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'**The patient will not have an adverse prognosis, with a probability of:** <span style="font-size:24px; font-weight:bold;">{prediction_proba[0][0]:.2f}</span>',
                    unsafe_allow_html=True
                )
        elif predict_result_code_FULL == 5:
            RP5_model = load_model(model_name='PhenoGroup5-RP')
            feature_names = ['Age', 'SBP', 'DBP', 'HB', 'AST', 'ALB', 'Urea', 'Creatinine', 'Cl','GLU','FS', 'LAD', 'RATD','RAD''Gender', 'Smoking_history', 'Hypertension', 'Hyperlipemia', 'Coronary_heart_disease', 'Auricular_fibrillation', 'Diabetes', 'Cerebral_stroke', 'Hypertensive_heart_disease']
            format_data_df = pd.DataFrame(data=[format_data], columns=feature_names)
            prediction = RP5_model.predict(format_data_df)
            prediction_proba = RP5_model.predict_proba(format_data_df)
            if prediction == 1:
                st.markdown(
                    f'**The patient will have an adverse prognosis, with a probability of:** <span style="font-size:24px; font-weight:bold;">{1-prediction_proba[0][0]:.2f}</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'**The patient will not have an adverse prognosis, with a probability of:** <span style="font-size:24px; font-weight:bold;">{prediction_proba[0][0]:.2f}</span>',
                    unsafe_allow_html=True
                )

