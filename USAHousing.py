from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('./USA_Housing.csv')

# sns.pairplot(df)
# sns.distplot(df['Price'])
# sns.heatmap(df.corr(), annot=True)

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population',]]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.intercept_)

cdf = pd.DataFrame(lm.coef_, X.columns)

predictions = lm.predict(X_test)
# plt.scatter(y_test, predictions)
sns.distplot((y_test-predictions))

print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print(cdf)
plt.show()