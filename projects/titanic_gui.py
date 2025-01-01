#importing necessary packages
import pandas as pd #for creating and handling dataframes
import streamlit as st #for gui
from streamlit_option_menu import option_menu# for option menu

st.title('ALGORITHMS')

data_eda=pd.read_csv(r'C:\Users\rajes\OneDrive\Desktop\titanic_dataset.csv') #for eda purposes
data_pda=pd.read_csv(r'C:\Users\rajes\OneDrive\Desktop\titanic_dataset.csv') #for pda purposes

st.info('[info] data loaded successfully...')

necessary_columns=['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked'] #creating a list of wanted columns
data_pda=data_pda[necessary_columns]  #updating the dataframe to contain only necessary columns

#replacing the NaN values in age columnns with mean of age
mean_age=data_pda['Age'].mean() #getting mean of age column
data_pda['Age']=data_pda['Age'].fillna(mean_age) #replacing NaN values with mean age value


#replacing the NaN values in embarked columnns with mode
embarked_mode=data_pda['Embarked'].mode() #getting mean of Embarked column
data_pda['Embarked']=data_pda['Embarked'].fillna('S') #replacing NaN values with mean age value

data_pda['Sex']=data_pda['Sex'].map({'male':1,'female':2})  #converting text labels to numbers
data_pda['Embarked']=data_pda['Embarked'].map({'C':1,'Q':2,'S':3})  #converting text labels to numbers

#segregating data into features and target
x_columns=['Pclass','Sex','Age','SibSp','Parch','Embarked'] #defining input columns
x=data_pda[x_columns].values  #choosing input columns

y=data_pda['Survived'].values  #choosing output columns

#splitting data into training and testing partitions
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

st.info('[info] data segregated and splitting complete...')

with st.sidebar:
    algo_selection=option_menu('SELECT A MODEL',options=['LOGISTIC REGRESSION','KNN','SVM','DECISION TREE','RANDOM FOREST','XG BOOST','NAIVE-BAYES'])

if algo_selection=='LOGISTIC REGRESSION':
    from sklearn.linear_model import LogisticRegression  # importing algo

    logreg_model = LogisticRegression(max_iter=15000)  # initialising the algo
    logreg_model.fit(x_train, y_train)

    # making the model predict answers for x_test
    y_pred = logreg_model.predict(x_test)

    # defining actual answers
    y_actual = y_test

    # comparing actual answers with predictions from the model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    st.info(f'the accuracy is:{accuracy_score(y_pred, y_actual)}')
    st.info(f'the precision is:{precision_score(y_pred, y_actual)}')
    st.info(f'the recall is:{recall_score(y_pred, y_actual)}')
    st.info(f'the f1 is:{f1_score(y_pred, y_actual)}')


if algo_selection=='KNN':
    from sklearn.neighbors import KNeighborsClassifier
    knn_model=KNeighborsClassifier() #initialising the model
    knn_model.fit(x_train,y_train)  # training the algorithm based on training dataset

    #making the model predict the answer
    knn_y_pred=knn_model.predict(x_test)

    # defining the actual answer
    knn_y_actual=y_test

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    st.info(f'the accuracy is:{accuracy_score(knn_y_pred, knn_y_actual)}')
    st.info(f'the precision is:{precision_score(knn_y_pred, knn_y_actual)}')
    st.info(f'the recall is:{recall_score(knn_y_pred, knn_y_actual)}')
    st.info(f'the f1 is:{f1_score(knn_y_pred, knn_y_actual)}')


if algo_selection=='SVM':
    from sklearn.svm import SVC
    svm_model=SVC() #initialising the model
    svm_model.fit(x_train,y_train) #training the algorithm based on training model

    svm_y_pred=svm_model.predict(x_test) #making the algorithm predict the answer
    svm_y_actual=y_test #defining the actual answer

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    st.info(f'the accuracy is:{accuracy_score(svm_y_pred, svm_y_actual)}')
    st.info(f'the precision is:{precision_score(svm_y_pred, svm_y_actual)}')
    st.info(f'the recall is:{recall_score(svm_y_pred, svm_y_actual)}')
    st.info(f'the f1 is:{f1_score(svm_y_pred, svm_y_actual)}')

if algo_selection=='NAIVE-BAYES':
    from sklearn.naive_bayes import GaussianNB

    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_train)
    print('[info] model training complete')
    gnb_y_pred = gnb_model.predict(x_test)
    gnb_y_actual = y_test

    # comparing actual answers with predictions from the model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    st.info(f'the accuracy is:{accuracy_score(gnb_y_pred, gnb_y_actual)}')
    st.info(f'the precision is:{precision_score(gnb_y_pred, gnb_y_actual)}')
    st.info(f'the recall is:{recall_score(gnb_y_pred, gnb_y_actual)}')
    st.info(f'the f1 is:{f1_score(gnb_y_pred, gnb_y_actual)}')


if algo_selection=='DECISION TREE':
    from sklearn.tree import DecisionTreeClassifier  # importing the algo
    dt_model=DecisionTreeClassifier()
    dt_model.fit(x_train,y_train)

    dt_y_pred=dt_model.predict(x_test)
    dt_y_actual=y_test

    # comparing actual answers with predictions from the model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    st.info(f'the accuracy is:{accuracy_score(dt_y_pred, dt_y_actual)}')
    st.info(f'the precision is:{precision_score(dt_y_pred, dt_y_actual)}')
    st.info(f'the recall is:{recall_score(dt_y_pred, dt_y_actual)}')
    st.info(f'the f1 is:{f1_score(dt_y_pred, dt_y_actual)}')


if algo_selection=='RANDOM FOREST':
    from sklearn.ensemble import RandomForestClassifier  # importing the algo
    rf_model=RandomForestClassifier()

    rf_model.fit(x_train,y_train)

    rf_y_pred=rf_model.predict(x_test)
    rf_y_actual=y_test

    # comparing actual answers with predictions from the model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    st.info(f'the accuracy is:{accuracy_score(rf_y_pred, rf_y_actual)}')
    st.info(f'the precision is:{precision_score(rf_y_pred, rf_y_actual)}')
    st.info(f'the recall is:{recall_score(rf_y_pred, rf_y_actual)}')
    st.info(f'the f1 is:{f1_score(rf_y_pred, rf_y_actual)}')