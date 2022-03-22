import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error,explained_variance_score


df = pd.read_csv('kc_house_data.csv')

# check for null values
# print(df.isnull().sum())

# see statistical mean std count...
# print(df.describe().transpose())

# sns.distplot(df['price'])
# sns.countplot(df['bedrooms'])

# Getting the correlation between features
# print(df.corr())

# sns.scatterplot(x='price', y='sqft_living', data=df)
# sns.boxplot(x='bedrooms', y='price', data=df)

# sns.scatterplot(x='price', y='long', data=df)

# Removing the top 1% houses because its not necessary
non_top_1_percent = df.sort_values('price', ascending=False).iloc[216:]

# seeing the distribution of the houses on the map
# sns.scatterplot(x='long', y='lat', data=non_top_1_percent, hue='price',edgecolor=None, alpha=.2, palette='RdYlGn')

# feature engineering
df = df.drop('id', axis=1)

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

# sns.boxplot(x='month', y='price', data=df)
# df.groupby('year').mean().price.plot()

df = df.drop('date', axis=1)
df = df.drop('zipcode',axis=1)

X = df.drop('price', axis=1).values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=101)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Here we can evaluate the model on the go
history = model.fit(x=X_train, y=y_train, validation_data = (X_test, y_test), batch_size=128, epochs=400)

model.evaluate(X_test, y_test, verbose=1)

loss_df = pd.DataFrame(history.history)
# loss_df.plot()

predictions = model.predict(X_test)

print(mean_squared_error(y_test,predictions))
print(mean_absolute_error(y_test,predictions))
print(explained_variance_score(y_test, predictions))
#The Predictions are off by 20% which is not good

plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')

# sns.scatterplot(x='Test True Y', y='Model Predictions',data=pred_df)


plt.show()