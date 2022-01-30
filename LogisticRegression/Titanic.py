import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')

# analyzing data

# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.set_style('whitegrid')
# sns.countplot(x='Survived', hue='Pclass', data=train, palette='RdBu_r')

# sns.distplot(train['Age'].dropna(), kde=False, bins=30)

# sns.countplot(x='SibSp', data=train)


# cleaning Data
# sns.boxplot(x='Pclass', y='Age', data=train)

def impute_age(cols):
    Age = cols[0]
    PClass = cols[1]

    if pd.isnull(Age):
        if PClass==1:
            return 37
        elif PClass==2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
test['Age'] = test[['Age', 'Pclass']].apply(impute_age, axis=1)
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)
test.dropna(inplace=True)
# sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#Dealing with categorical columns
sexTrain = pd.get_dummies(train['Sex'], drop_first=True)
embarkTrain = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sexTrain, embarkTrain], axis=1)

sexTest =  pd.get_dummies(test['Sex'], drop_first=True)
embarkTest = pd.get_dummies(test['Embarked'], drop_first=True)
test = pd.concat([test, sexTest, embarkTest], axis=1)
#drop unwanted
train.drop(['Sex','Embarked','Name','Ticket', 'PassengerId'], axis=1, inplace=True)

# Splitting to train test split
X_train= train.drop('Survived', axis=1)
y_train= train['Survived']

X_test= test.drop('Survived', axis=1)
y_test= test['Survived']

# Or You could use the train test split like the last one
# X= train.drop('Survived', axis=1)
# y= train['Survived']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

plt.show()

