#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:11:15 2021

@author: shoeb
"""

from flask import Flask,render_template,request,redirect,url_for
import pandas as pd
from warnings import simplefilter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



simplefilter(action='ignore', category = FutureWarning)

app = Flask(__name__,static_url_path='/static')


'''
    # try_data=[[63,1,1,145,233,1,2,150,0,2.3,3,0,6]]
    # t1=[[67,1,4,120,229,0,2,129,1,2.6,2,2,7]]
    return y
'''
@app.route('/',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def Home():
    #try_data=[[63,1,1,145,233,1,2,150,0,2.3,3,0,6]]
    #s1=str(try_data)
    #out ="<h1>HEllo WORLD!</h1><p>"+s1+"</p>"
    
    
    if(request.method=='POST'):
        age=int(request.form['age'])
        gender=int(request.form['gender'])
        chest=int(request.form['chest'])
        bp=int(request.form['bp'])
        cl=int(request.form['cl'])
        fbs=int(request.form['fbs'])
        ecg=int(request.form['ecg'])
        mha=int(request.form['mha'])
        eia=int(request.form['eia'])
        dep=float(request.form['dep'])
        ST=int(request.form['ST'])
        ves=float(request.form['ves'])
        thal=int(request.form['thal'])
        
        data=[age,gender,chest,bp,cl,fbs,ecg,mha,eia,dep,ST,ves,thal]
        
        #ML CODE
        df = pd.read_csv('cleveland.csv', header = None)
        df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                      'fbs', 'restecg', 'thalach', 'exang', 
                      'oldpeak', 'slope', 'ca', 'thal', 'target']
        df.isnull().sum()
        df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
        df['sex'] = df.sex.map({0: 'female', 1: 'male'})
        df['thal'] = df.thal.fillna(df.thal.mean())
        df['ca'] = df.ca.fillna(df.ca.mean())
        df['sex'] = df.sex.map({'female': 0, 'male': 1})
    
        
        ################### data preprocessing   ###################################
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        
        sc = ss()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        
        ############################################# Classification ALgorithms  #############################################################
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
       
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
        
        ###########   Classifier_1 Logistic Regression    #############
        classifier1 = LogisticRegression(solver='lbfgs', max_iter=1000)
        classifier1.fit(X_train, y_train)
        x = classifier1.predict([data])
        for j in x:
                y1=j
        
        ###############  Classifier_2 SVM ##########
        classifier2 = SVC(kernel = 'rbf')
        classifier2.fit(X_train, y_train)
        x = classifier2.predict([data])
        for j in x:
                y2=j
        
        ############### Classifier_3  Naive Bayes  ##########
        classifier3 = GaussianNB()
        classifier3.fit(X_train, y_train)
        x = classifier3.predict([data])
        for j in x:
                y3=j
        
        
        ################# Classifier 4  Dicision Tree  #############
        classifier4 = DecisionTreeClassifier()
        classifier4.fit(X_train, y_train)
        x = classifier4.predict([data])
        for j in x:
                y4=j
        
        ############## Classifier 5  Random Forest   #################
        classifier5 = RandomForestClassifier(n_estimators = 10)
        classifier5.fit(X_train, y_train)
        x = classifier5.predict([data])
        for j in x:
                y5=j
        
        
        ###################Hybrid Answer  #####################
       
        
        y=y1+y2+y3+y4+y5
        if(y>=3):
            y=1
        else:
            y=0
        #Result_list = [y1,y2,y3,y4,y5,y]
        ##############################################################END##############################################
                
        #SEND VALUE
        return redirect(url_for('result',resultant = y,c1=y1,c2=y2,c3=y3,c4=y4,c5=y5))
    
    return render_template('home.html')

@app.route('/result',methods=['GET','POST'])
def result():
      res = request.args.get("resultant")
      c1= request.args.get("c1")
      c2= request.args.get("c2")
      c3= request.args.get("c3")
      c4= request.args.get("c4")
      c5= request.args.get("c5")
      return render_template('result.html',res = res,c1=c1,c2=c2,c3=c3,c4=c4,c5=c5)

if __name__ == '__main__':
    app.run(debug=True)
