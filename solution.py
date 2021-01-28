import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

bank = pd.read_csv("bank-full.csv",sep=';')
bank = bank[["age", "balance", "housing", "loan", "duration","campaign", "pdays", "previous", "y"]]

le = LabelEncoder();
bank["housing"] = le.fit_transform(bank["housing"])
bank["loan"] = le.fit_transform(bank["loan"])
bank["y"] = le.fit_transform(bank["y"])

st.write("""
# Protótipo Grupo 31
## SINO-PL MiEGSI 2020/2021
""")

st.header("Dataset")
st.write(bank)

st.sidebar.header("Input")

def user_input():
    #age
    age_input = st.sidebar.slider("Idade", 18, 95, 41)
    #balance
    balance_input = st.sidebar.slider("Saldo", -8019, 102127, 1362)
    #housing
    housing_input = st.sidebar.checkbox("Tem empréstimo à habitação?")
    #loan
    loan_input = st.sidebar.checkbox("Tem empréstimo pessoal?")
    #duration
    duration_input = st.sidebar.slider("Duração último contacto", 0, 4918, 258)
    #campaign
    campaign_input = st.sidebar.slider("Contactos nesta campanha", 1, 63, 3)
    #pdays
    pdays_input = st.sidebar.slider("Dias desde último contacto", 0, 871, 40)
    #previous
    previous_input = st.sidebar.slider("Nr. contactos anteriores", 0, 275, 1)

    data = {"age": age_input,
            "balance": balance_input,
            "housing": housing_input,
            "loan": loan_input,
            "duration": duration_input,
            "campaign": campaign_input,
            "pdays": pdays_input,
            "previous": previous_input}
    user_input = pd.DataFrame(data, index=[0])
    return user_input

ui = user_input();

st.header("Input do utilizador")
st.write(ui)

X = bank[["age", "balance", "housing", "loan", "duration","campaign", "pdays", "previous"]]
y = bank["y"]

lr = LogisticRegression()
lr.fit(X, y)
predict = lr.predict(ui)
predict_prob = lr.predict_proba(ui)

st.subheader("Previsão")
st.write(predict)

st.subheader("Probabilidade da previsão")
st.write(predict_prob)
