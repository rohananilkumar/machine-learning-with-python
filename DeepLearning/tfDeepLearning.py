import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv('fake_reg.csv')

X = df[['feature1', 'feature2']].values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.33, random_state=42)

# print(df.head())

sns.pairplot(df)


plt.show()