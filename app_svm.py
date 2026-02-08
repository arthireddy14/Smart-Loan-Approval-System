"""
Smart Loan Approval System
--------------------------
An end-to-end machine learning application that predicts loan approval
using Support Vector Machines (Linear, Polynomial, RBF).

Tech Stack:
- Python
- Scikit-Learn
- Streamlit
- Pandas, NumPy

Features:
- Data preprocessing and feature scaling
- Kernel comparison
- Interactive UI for real-time prediction
"""

import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score


# Page config

st.set_page_config("SVM-RBF Loan Approval",layout="centered")

# Loading CSS file
def load_css(file):
    with open (file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
        
load_css("styles.css")

# Title of the app
st.markdown("""
<div class="card">
<h1>Smart Loan Approval System </h1>
<p>This system uses Support Vector Machines to predict loan approval.</p>
</div>
""",unsafe_allow_html=True)


# sidebar inputs
income=st.sidebar.number_input("Applicant income ",min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount ", min_value=0)

credit = st.sidebar.radio("Credit History", ["Yes", "No"])
credit = 1 if credit == "Yes" else 0

employment = st.sidebar.selectbox("Employment Status ", ["No", "Yes"])
property_area = st.sidebar.selectbox("Property Area ", ["Urban", "Semiurban", "Rural"])
if st.button("Description"):
    st.markdown("""<div class="card">
                <p>What does this system do?  
                This application predicts whether a loan is likely to be approved based on income, loan amount, credit history, employment status, and property area.
                How does it work?  
                It uses a machine learning model called Support Vector Machine (SVM), which learns from past loan data to identify patterns between approved and rejected applications.
                Why is this useful?   
                It helps financial institutions make faster and more consistent loan decisions while reducing human bias.</p></div>""",unsafe_allow_html=True)
    
# Encoding the inputs
emp_yes = 1 if employment == "Yes" else 0
semiurban = 1 if property_area == "Semiurban" else 0
urban = 1 if property_area == "Urban" else 0

# Loading and preprocessing Loan dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

df = df[['ApplicantIncome','LoanAmount','Credit_History',
         'Self_Employed','Property_Area','Loan_Status']]

df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Credit_History'].fillna(0, inplace=True)
df['Self_Employed'].fillna('No', inplace=True)

df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0})
df = pd.get_dummies(df, drop_first=True)

x = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Scaler-Standardize numerical features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Match input features
feature_names=df.drop('Loan_Status',axis=1).columns
input_data = pd.DataFrame([[income, loan_amount, credit, emp_yes, semiurban, urban]],
                          columns=feature_names)
input_scaled = scaler.transform(input_data)

# Model selection 
kernel=st.radio("Select SVM Kernel ",["Linear SVM","Polynomial SVM","RBF SVM"])

# Train SVM model with selected kernel

if kernel == "Linear SVM":
    model = SVC(kernel='linear', probability=True)
elif kernel == "Polynomial SVM":
    model = SVC(kernel='poly', degree=3, probability=True)
else:
    model = SVC(kernel='rbf', probability=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)


# Visualization
st.subheader("Exploaratory Data Analysis")
fig,ax=plt.subplots()
sns.countplot(x='Loan_Status',data=df,ax=ax)
st.pyplot(fig)
# Performance

results={
    "Linear":SVC(kernel='linear').fit(x_train,y_train).score(x_test,y_test),
    "Polynomial":SVC(kernel='poly').fit(x_train,y_train).score(x_test,y_test),
    "RBF":SVC(kernel='rbf').fit(x_train,y_train).score(x_test,y_test)
}
st.bar_chart(results)


# Predict loan eligibility 

if st.button("Check Loan Eligibility"):
    if income<=0 or loan_amount<=0:
        st.warning("Please eneter valid income and loan amount.")
    else:
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0][1]
        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    st.metric("Model Accuracy ",f"{accuracy*100:.2f}%")
    st.write("Precison : " f"{precision:.2f}")
    st.write("Recall :" f"{recall:.2f}")
    st.write("F1-Score : " f"{f1:.2f}")
    st.write(f"**Kernel Used:** {kernel}")

    st.info(
        "Based on credit history and income pattern, "
        "the applicant is likely / unlikely to repay the loan."
    )