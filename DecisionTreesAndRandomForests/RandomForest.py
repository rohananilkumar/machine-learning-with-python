import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('kyphosis.csv')
print(df.head())

#sns.pairplot(df, hue='Kyphosis')

X=df.drop('Kyphosis', axis=1)
y=df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# plt.show()
