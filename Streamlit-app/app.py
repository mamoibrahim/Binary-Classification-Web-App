from sklearn import model_selection
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay , confusion_matrix
from sklearn.metrics import precision_score, recall_score 
from  sklearn.svm import SVC 
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown('Are your mushrooms edible or poisonous? 🍄')
    st.sidebar.markdown('Are your mushrooms edible or poisonous? 🍄')

    df = pd.read_csv("mushrooms.csv")
    df = df.replace('?', np.nan)
    
    @st.cache_data(persist=True)
    def load_data():  
        
        label = LabelEncoder()
        for col in df.columns:
             df[col] = label.fit_transform(df[col])
        return df
    df = load_data()
    


    
    
    @st.cache_data(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    x_train, x_test, y_train, y_test = split(df) 
   

    def plot_metrics(metrics_list , y_test , y_pred ):
        
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            display_labels=["edible", "poisonous"]
            cm = confusion_matrix(y_test, y_pred)
            disp =  ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            disp.plot()
            st.pyplot()

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            disp1 = RocCurveDisplay.from_estimator(model, x_test, y_test, name="SVM")
            disp1.plot()
            st.pyplot()

        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            disp2 = PrecisionRecallDisplay.from_estimator(model,x_test, y_test, name="SVM" )
            disp2.plot()
            st.pyplot() 

           
    

    if st.sidebar.checkbox("Show raw data", False) : 
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
    


    st.sidebar.subheader("Choose a Classifier")
    classifier = st.sidebar.selectbox("", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_SVM")
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel_SVM")
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma_SVM")

        metrics = st.sidebar.multiselect("Select Metrics", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key="metrics_SVM")
        if st.sidebar.button("Classify", key="classify_SVM"):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=["edible", "poisonous"]))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=["edible", "poisonous"]))
            plot_metrics(metrics, y_test, y_pred)
        

    elif classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input('C (Regularization parameter)', 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter_LR")

        metrics = st.sidebar.multiselect('Select Metrics' , ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'), key='metrics_LR')

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=["edible", "poisonous"]))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=["edible", "poisonous"]))
            plot_metrics(metrics, y_test, y_pred)

    elif classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key="n_estimators")
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False), key="bootstrap")

        metrics = st.sidebar.multiselect("Select Metrics", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key="metrics_RF")

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", precision_score(y_test, y_pred, labels=["edible", "poisonous"]))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=["edible", "poisonous"]))
            plot_metrics(metrics, y_test, y_pred) 



    

main()