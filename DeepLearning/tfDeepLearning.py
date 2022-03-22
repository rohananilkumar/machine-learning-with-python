from tkinter.filedialog import test
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model


df = pd.read_csv('fake_reg.csv')

X = df[['feature1', 'feature2']].values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, random_state=42)

# print(df.head())

#normalizing the data
scaler = MinMaxScaler()
scaler.fit(X_test)

#Normalizing the training and test set
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# sns.pairplot(df)

# models = Sequential([Dense(4, activation='relu'), Dense(2, activation='relu'), Dense(1)])

# or

model = Sequential()
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')

model.fit(x=X_train, y=y_train, epochs=250)
# epoch = number of times the training algorithm will go through the traing data
# At a later stage, we'll be able to make the model stop automatically when a certain stage of optimization is reached

loss_df = pd.DataFrame(model.history.history)
# loss_df.plot()

model.evaluate(X_test, y_test, verbose=1)

test_predictions = model.predict(X_test)

# test predictions
print(test_predictions)
test_predictions = pd.Series(test_predictions.reshape(300,))
print(test_predictions)

pred_df = pd.DataFrame(y_test, columns=['Test True Y'])
pred_df = pd.concat([pred_df,test_predictions], axis=1)
pred_df.columns = ['Test True Y', 'Model Predictions']

sns.scatterplot(x='Test True Y', y='Model Predictions',data=pred_df)

# Evaluating the model
print(mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions']))
print(mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions']))

# Predicting on a new gem

new_gem = [[998, 1000]]

new_gem = scaler.transform(new_gem)

prediction = model.predict(new_gem)

# Saving the model to a file
model.save('my_gem_model.h5')

# Loading the model from a file
later_model = load_model('my_gem_model.h5')

later_model.predict(new_gem)

plt.show()