#importing necessary packages
import numpy as np #array ops
import pandas as pd#handling dataframes
import matplotlib.pyplot as plt #for visualization
import seaborn as sns #for advanced vis.
import streamlit as st #for gui webpage
from streamlit_option_menu import option_menu# for option menu


        #1.DATA COLLECTION
data=pd.read_csv(r'C:\Users\rajes\OneDrive\Desktop\diabetes.csv')
st.info('[info] data loaded successfully')

        #2.DATA PRE-PROCESSING

st.info(data.columns[data.isna().any()])

#feature selection and feature engineering
#segregation
x=data.iloc[:,:-1].values#choosing features
y=data.iloc[:,-1].values#choosing targets

st.info('[info] data segregated into features and targets successfully')

#splitting into training and testing partitions
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=44)
st.info('[info] data segregated into training and testing set successfully')

        #BUILDING THE GUI

st.title('INTEGRATING GUI WITH ML MODELS')#setting title of webpage

#taking new user inputs
pregnancies=st.number_input('ENTER NO. OF PREGNANCIES',step=1)
glucose=st.number_input('ENTER BLOOD GLUCOSE',step=1)
blood_pressure=st.number_input('ENTER BLOOD PRESSURE:',step=1)
skin_thickness=st.number_input('ENTER SKIN THICKNESS',step=1)
insulin=st.number_input('ENTER INSULIN',step=1)
bmi=st.number_input('ENTER BMI')
diabetes_pedigree=st.number_input('ENTER DIABETES PEDIGREE')
age=st.number_input('ENTER PATIENT AGE',step=1,min_value=5,max_value=100)

#storing new user input into a 2d array
new_user_input=[[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree,age]]


with st.sidebar:#adding option menu to sidebar  
    model_selection=option_menu('SELECT A CLASSIFICATION MODEL',options=['LOGISTIC REGRESSION','KNN','SVM','DECISION TREE','RANDOM FOREST','XG BOOST','NAIVE-BAYES'])


if model_selection=='LOGISTIC REGRESSION':
    #3a.MODEL TRAINING : LOGISTIC REGRESSION
    from sklearn.linear_model import LogisticRegression #importing the algo
    logistic_regression_model=LogisticRegression(max_iter=1500) #initialising the model
    logistic_regression_model.fit(x_train,y_train) #training logistic regression on preprocessed data
    #4a. MODEL EVALUATION : LOGISTIC REGRESSION
    #making the model predict ouptut for x_test
    logistic_regression_model_predicted=logistic_regression_model.predict(x_test)
    #defining the actual values
    logistic_regression_model_actual=y_test
    #comparing actual and predicted values to get model parameters

    st.subheader('MODEL PARAMETERS')
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(logistic_regression_model_actual, logistic_regression_model_predicted)
    st.info(f'accuracy of the model is:{accuracy}')

    precision = precision_score(logistic_regression_model_actual, logistic_regression_model_predicted)
    st.error(f'precision of the model is: {precision}')

    recall = recall_score(logistic_regression_model_actual, logistic_regression_model_predicted)
    st.success(f'recall of the model is:{recall}')

    f1 = f1_score(logistic_regression_model_actual, logistic_regression_model_predicted)
    st.warning(f'f1 score of the model is: {f1}')

    st.subheader('MODEL DIAGNOSIS')
    logistic_model_output=logistic_regression_model.predict(new_user_input)
    if logistic_model_output[0]==1:
        st.error('THE PATIENT IS DIABETIC')
    if logistic_model_output[0]==0:
        st.success('THE PATIENT IS NON-DIABETIC')

if model_selection=='KNN':
    st.info('this is KNN algo')

    #3b.MODEL TRAINING : knn
    from sklearn.neighbors import KNeighborsClassifier #importing the algo
    knn_model=KNeighborsClassifier() #initialising the model
    knn_model.fit(x_train,y_train) #training knn  on preprocessed data
    #4b. MODEL EVALUATION : knn 
    #making the model predict ouptut for x_test
    knn_model_predicted=knn_model.predict(x_test)
    #defining the actual values
    knn_model_actual=y_test
    #comparing actual and predicted values to get model parameters

    st.subheader('MODEL PARAMETERS')
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(knn_model_actual, knn_model_predicted)
    st.info(f'accuracy of the model is:{accuracy}')

    precision = precision_score(knn_model_actual, knn_model_predicted)
    st.error(f'precision of the model is: {precision}')

    recall = recall_score(knn_model_actual, knn_model_predicted)
    st.success(f'recall of the model is:{recall}')

    f1 = f1_score(knn_model_actual, knn_model_predicted)
    st.warning(f'f1 score of the model is: {f1}')

    st.subheader('MODEL DIAGNOSIS')
    knn_model_output=knn_model.predict(new_user_input)
    if knn_model_output[0]==1:
        st.error('THE PATIENT IS DIABETIC')
    if knn_model_output[0]==0:
        st.success('THE PATIENT IS NON-DIABETIC')


if model_selection=='SVM':
    st.success('this is Support Vector Machine')
    
    #3c.MODEL TRAINING : svm
    from sklearn.svm import SVC #importing the algo
    svm_model=SVC() #initialising the model
    svm_model.fit(x_train,y_train) #training svm  on preprocessed data
    #4c. MODEL EVALUATION : svm 
    #making the model predict ouptut for x_test
    svm_model_predicted=svm_model.predict(x_test)
    #defining the actual values
    svm_model_actual=y_test
    #comparing actual and predicted values to get model parameters

    st.subheader('MODEL PARAMETERS')
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(svm_model_actual, svm_model_predicted)
    st.info(f'accuracy of the model is:{accuracy}')

    precision = precision_score(svm_model_actual, svm_model_predicted)
    st.error(f'precision of the model is: {precision}')

    recall = recall_score(svm_model_actual, svm_model_predicted)
    st.success(f'recall of the model is:{recall}')

    f1 = f1_score(svm_model_actual, svm_model_predicted)
    st.warning(f'f1 score of the model is: {f1}')

    st.subheader('MODEL DIAGNOSIS')
    svm_model_output=svm_model.predict(new_user_input)
    if svm_model_output[0]==1:
        st.error('THE PATIENT IS DIABETIC')
    if svm_model_output[0]==0:
        st.success('THE PATIENT IS NON-DIABETIC')

if model_selection=='DECISION TREE': 
    #3c.MODEL TRAINING : decisiontree
    from sklearn.tree import DecisionTreeClassifier #importing the algo
    decisiontree_model=DecisionTreeClassifier() #initialising the model
    decisiontree_model.fit(x_train,y_train) #training decisiontree  on preprocessed data
    #4c. MODEL EVALUATION : decisiontree 
    #making the model predict ouptut for x_test
    decisiontree_model_predicted=decisiontree_model.predict(x_test)
    #defining the actual values
    decisiontree_model_actual=y_test
    #comparing actual and predicted values to get model parameters

    st.subheader('MODEL PARAMETERS')
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(decisiontree_model_actual, decisiontree_model_predicted)
    st.info(f'accuracy of the model is:{accuracy}')

    precision = precision_score(decisiontree_model_actual, decisiontree_model_predicted)
    st.error(f'precision of the model is: {precision}')

    recall = recall_score(decisiontree_model_actual, decisiontree_model_predicted)
    st.success(f'recall of the model is:{recall}')

    f1 = f1_score(decisiontree_model_actual, decisiontree_model_predicted)
    st.warning(f'f1 score of the model is: {f1}')
    
    st.subheader('MODEL DIAGNOSIS')
    decisiontree_model_output=decisiontree_model.predict(new_user_input)
    if decisiontree_model_output[0]==1:
        st.error('THE PATIENT IS DIABETIC')
    if decisiontree_model_output[0]==0:
        st.success('THE PATIENT IS NON-DIABETIC')


if model_selection=='RANDOM FOREST':
    #3d.MODEL TRAINING : random_forest
    from sklearn.ensemble import RandomForestClassifier #importing the algo
    random_forest_model=RandomForestClassifier() #initialising the model
    random_forest_model.fit(x_train,y_train) #training random_forest  on preprocessed data
    #4d. MODEL EVALUATION : random_forest 
    #making the model predict ouptut for x_test
    random_forest_model_predicted=random_forest_model.predict(x_test)
    #defining the actual values
    random_forest_model_actual=y_test
    #comparing actual and predicted values to get model parameters

    st.subheader('MODEL PARAMETERS')
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(random_forest_model_actual, random_forest_model_predicted)
    st.info(f'accuracy of the model is:{accuracy}')

    precision = precision_score(random_forest_model_actual, random_forest_model_predicted)
    st.error(f'precision of the model is: {precision}')

    recall = recall_score(random_forest_model_actual, random_forest_model_predicted)
    st.success(f'recall of the model is:{recall}')

    f1 = f1_score(random_forest_model_actual, random_forest_model_predicted)
    st.warning(f'f1 score of the model is: {f1}')
    st.subheader('MODEL DIAGNOSIS')
    random_forest_model_output=random_forest_model.predict(new_user_input)
    if random_forest_model_output[0]==1:
        st.error('THE PATIENT IS DIABETIC')
    if random_forest_model_output[0]==0:
        st.success('THE PATIENT IS NON-DIABETIC')


if model_selection=='NAIVE-BAYES':
    # 3d.MODEL TRAINING : naive_bayes
    from sklearn.naive_bayes import GaussianNB  # importing the algo

    naive_bayes_model = GaussianNB()  # initialising the model
    naive_bayes_model.fit(x_train, y_train)  # training naive_bayes  on preprocessed data
    # 4d. MODEL EVALUATION : naive_bayes
    # making the model predict ouptut for x_test
    naive_bayes_model_predicted = naive_bayes_model.predict(x_test)
    # defining the actual values
    naive_bayes_model_actual = y_test
    # comparing actual and predicted values to get model parameters

    st.subheader('MODEL PARAMETERS')
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(naive_bayes_model_actual, naive_bayes_model_predicted)
    st.info(f'accuracy of the model is:{accuracy}')

    precision = precision_score(naive_bayes_model_actual, naive_bayes_model_predicted)
    st.error(f'precision of the model is: {precision}')

    recall = recall_score(naive_bayes_model_actual, naive_bayes_model_predicted)
    st.success(f'recall of the model is:{recall}')

    f1 = f1_score(naive_bayes_model_actual, naive_bayes_model_predicted)
    st.warning(f'f1 score of the model is: {f1}')

    st.subheader('MODEL DIAGNOSIS')
    naive_bayes_model_output = naive_bayes_model.predict(new_user_input)
    if naive_bayes_model_output[0] == 1:
        st.error('THE PATIENT IS DIABETIC')
    if naive_bayes_model_output[0] == 0:
        st.success('THE PATIENT IS NON-DIABETIC')
        

if model_selection=='XG BOOST':
    # 3d.MODEL TRAINING : xg_boost
    from xgboost import XGBClassifier  # importing the algo

    xg_boost_model = XGBClassifier()  # initialising the model
    xg_boost_model.fit(x_train, y_train)  # training xg_boost  on preprocessed data
    # 4d. MODEL EVALUATION : xg_boost
    # making the model predict ouptut for x_test
    xg_boost_model_predicted = xg_boost_model.predict(x_test)
    # defining the actual values
    xg_boost_model_actual = y_test
    # comparing actual and predicted values to get model parameters

    st.subheader('MODEL PARAMETERS')
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(xg_boost_model_actual, xg_boost_model_predicted)
    st.info(f'accuracy of the model is:{accuracy}')

    precision = precision_score(xg_boost_model_actual, xg_boost_model_predicted)
    st.error(f'precision of the model is: {precision}')

    recall = recall_score(xg_boost_model_actual, xg_boost_model_predicted)
    st.success(f'recall of the model is:{recall}')

    f1 = f1_score(xg_boost_model_actual, xg_boost_model_predicted)
    st.warning(f'f1 score of the model is: {f1}')

    st.subheader('MODEL DIAGNOSIS')
    xg_boost_model_output = xg_boost_model.predict(new_user_input)
    if xg_boost_model_output[0] == 1:
        st.error('THE PATIENT IS DIABETIC')
    if xg_boost_model_output[0] == 0:
        st.success('THE PATIENT IS NON-DIABETIC')