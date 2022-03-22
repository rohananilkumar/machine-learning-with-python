from gc import callbacks
from tabnanny import verbose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error,explained_variance_score
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_csv("cancer_classification.csv")

sns.countplot(x='benign_0__mal_1',data=df)

X = df.drop('benign_0__mal_1', axis=1).values
y=df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=101)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))

# Binary Classification
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

# Using early stopping to stop prevent over fitting to the data
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
# patience = number of epochs that should satisfy the condition... esentially for ignoring noise in the data

history = model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])

losses = pd.DataFrame(history.history)

losses.plot()


plt.show()


