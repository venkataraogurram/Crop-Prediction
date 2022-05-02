import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
import pickle

crop=pd.read_csv('Crop_recommendation.csv')
features=crop[['N','P','K','temperature','humidity','ph','rainfall']]
target=crop['label']
labels=crop['label']
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(features,target,test_size=0.2,random_state=2)

'''from sklearn.naive_bayes import GaussianNB
NaiveBayes=GaussianNB()
NaiveBayes.fit(Xtrain,Ytrain)

pickle.dump(NaiveBayes,open('model.pkl','wb'))'''
from sklearn.linear_model import LogisticRegression
LogisticReg=LogisticRegression()
LogisticReg.fit(Xtrain,Ytrain)


pickle.dump(LogisticReg,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[112,46,38,29,56,7,267]]))