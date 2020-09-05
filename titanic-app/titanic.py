#!/usr/bin/env python
# coding: utf-8



import pickle
import pandas as pd
import seaborn as sns
sns.set()

train=pd.read_csv("Dataset/train.csv")
test=pd.read_csv("Dataset/test.csv")

train.drop(['Cabin'],axis=1,inplace=True)
test.drop(['Cabin'],axis=1,inplace=True)


train_test_data=[train,test]
for dataset in train_test_data:
    dataset['Title']=dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 4, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)

train.drop(['Name'],axis=1,inplace=True)
test.drop(['Name'],axis=1,inplace=True)

sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

train.drop(['Ticket'],axis=1,inplace=True)
test.drop(['Ticket'],axis=1,inplace=True)

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)    

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

train.drop(['SibSp','Parch'],axis=1,inplace=True)
test.drop(['SibSp','Parch'],axis=1,inplace=True)

train["Age"].fillna(train.groupby(["Title",'Pclass'])["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby(["Title"])["Age"].transform("median"), inplace=True)
train.groupby(["Title",'Pclass'])["Age"].transform("median")

test.drop(['PassengerId'],inplace=True,axis=1)
print(test.head())

X = train.iloc[:, 2:].values 
Y = train.iloc[:, 1].values 

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini', 
                                           n_estimators=1100,
                                           max_depth=5,
                                           min_samples_split=4,
                                           min_samples_leaf=5,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=-1,
                                           verbose=1)
forest.fit(X, Y)

filename = 'model.pkl'
pickle.dump(forest, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X, Y)
print('Accuracy: %.2f' %(result * 100) + '%')




