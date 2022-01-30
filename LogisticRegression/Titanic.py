import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')

# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.set_style('whitegrid')
# sns.countplot(x='Survived', hue='Pclass', data=train, palette='RdBu_r')

# sns.distplot(train['Age'].dropna(), kde=False, bins=30)

sns.countplot(x='SibSp', data=train)

plt.show()

