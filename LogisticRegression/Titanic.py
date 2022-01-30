import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

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

# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


plt.show()

