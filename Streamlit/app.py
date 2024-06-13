import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_ydata_profiling import st_profile_report
from ydata_profiling import ProfileReport
from streamlit_option_menu import option_menu
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
import os

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded")

# Function to tranform the data and to predict the outcomes of test data
def predict():
    column_transformer = ColumnTransformer(transformers=[
    ('num', KNNImputer(),['Tenure','Complain', 'OrderCount', 'CashbackAmount'] ),
    ("onehot", encoder, ['PreferedOrderCat','MaritalStatus'])])

    pipeline = Pipeline([('preprocessing', column_transformer)])

    transformed_data = pd.DataFrame(pipeline.fit_transform(df),columns=model.feature_names_in_)
    y_pred = pd.DataFrame(model.predict(transformed_data),columns=['Churn'])
    pred = pd.concat([df, y_pred], axis=1)
    return pred

# Function to plot gauge graphs  
def plot_gauge(
    indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound,Threshold
):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number+delta",
            domain={"x": [0, 1], "y": [0, 1]},
            delta = {'reference':Threshold },
            number={
                "suffix": indicator_suffix,
                "font.size": 26,
            },
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1},
                "bar": {"color": indicator_color},
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': Threshold},
            },
            title={
                "text": indicator_title,
                "font": {"size": 28},
            },
        )
    )
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)


# Function to show metric KPIs
def plot_metric(label, value, prefix="", suffix=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 28,
            },
            title={
                "text": label,
                "font": {"size": 24},
            },
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# Function for bar graph
def plot_bar(data,X,Y,Z,Title):
    fig = px.bar(data,x=X,y=Y,color=Z,text_auto=".2s",title=Title,height=400,orientation='v')
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

# Function to plot scatter
def plot_scatter(data,X,Y,Z,Title):
    fig = px.scatter(data,x=X,y=Y,color=Z,title=Title)
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

# Function to for histogram
def plot_hist(data,X,Y,Z,Title):
    fig = px.histogram(data,x=X,y=Y,color=Z,title=Title)
    st.plotly_chart(fig, use_container_width=True)

# Function to plot pie graph

def plot_pie(data,X,Y,color,Title):
    fig = px.pie(data, values=X, names=Y,color=color, title=Title)
    st.plotly_chart(fig, use_container_width=True)

model = joblib.load(open('dtclassifier.joblib', 'rb'))
encoder = joblib.load(open('Encoder.joblib', 'rb'))

with st.sidebar:
    st.title('__Customer Churn Predictor__')
    choice = option_menu(menu_title=None,
                         options=["Individual Prediction","New Data Prediction", "Automated EDA","Dashboard"])
    st.info("This Webapp will automate the end-to-end process of building a machine learning model. Just provide the necessary inputs in the above given navigation pages.")

if os.path.exists("dataset.csv"):
    df = pd.read_csv('dataset.csv', index_col=None)

if choice == "Individual Prediction":
    st.title('Customer Churn Predictor')

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            Tenure=st.number_input("Tenure",value=None, placeholder="Select the Tenure of the Customer")
            OrderCount=st.number_input("Order Count",value=None, placeholder="Select the Order Count of the Customer")
            CashbackAmount=st.number_input("Cashback Amount",value=None, placeholder="Select the Cashback amount of the Customer")

        with c2:
            Complain = st.selectbox("Complains",("Yes","No"),index=None,placeholder="Select whether if the Customer had any complains")
            PreferedOrderCat = st.selectbox("Prefered Order Catergory",('Laptop & Accessory', 'Mobile Phone', 
                                            'Others', 'Fashion','Grocery'),index=None,placeholder="Select the Prefered Order Catergory of the Customer")
            MaritalStatus = st.selectbox("Marital Status",('Single', 'Divorced', 'Married'),index=None,placeholder="Select the Marital Status of the Customer")

    if Complain == "Yes":
        Complain=1
    else:
        Complain=0

    if st.button("Predict"):
        inputs = pd.DataFrame(np.array([Tenure, Complain, OrderCount, CashbackAmount,PreferedOrderCat , MaritalStatus]).reshape(1, -1),
                                                columns=['Tenure', 'Complain', 'OrderCount', 'CashbackAmount','PreferedOrderCat','MaritalStatus'])
        encoded = encoder.transform(inputs[['PreferedOrderCat','MaritalStatus']])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
        new_df = pd.concat([inputs.drop(columns=['PreferedOrderCat','MaritalStatus']), encoded_df], axis=1)
    
        prediction = model.predict(new_df)

        if prediction == 0:
            st.success('The Customer is ulikely to churn', icon="✅")
        else:
            st.error("The customer is likely to churn", icon="🚨")

if choice == "New Data Prediction":
    st.title('Import the Dataset')
    test = st.file_uploader("__Note:__ The test dataset should NOT contain the __target column__")

    if test:
        data = pd.read_csv(test, index_col=None)
        st.success('File Upload Successfully')

        if st.button("Predict"):
            pred = predict()
            st.dataframe(pred)
            st.download_button("Download Dataset", pred.to_csv(), "predicted.csv")

if choice == "Automated EDA":

    pred=predict()

    st.title("Exploratory Data Analysis")
    if st.button("Generate Profile Report"):
        profile=ProfileReport(pred)
        st_profile_report(profile)


if choice == "Dashboard":
    data=predict()
    top_left_column, top_right_column = st.columns((1, 5))
    bottom_left_column, bottom_right_column = st.columns(2)

    with top_right_column:
        column_1, column_2= st.columns(2)

        with column_1:
            plot_pie(data,None,"Churn","Churn","Title")

        with column_2:
            plot_bar(data,"Tenure","Churn","Churn",Title="AAAAAAAAAAA")

    with top_left_column:
        plot_metric("Customer Count", 4821, prefix="", suffix="")
        plot_gauge(
        85.4,"#0068C9", "%", "Current Ratio", 100,83.2
        )

    with bottom_left_column:
        plot_hist(data,"Tenure","Churn",None,Title="AAAAAAAAAAA")
    
    with bottom_right_column:
            plot_scatter(data,"OrderCount","CashbackAmount","Churn","Title")
