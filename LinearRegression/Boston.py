from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

boston = load_boston()

X = pd.DataFrame(boston['data'], columns= boston['feature_names'])
y = boston['target']

sns.heatmap(X.corr())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

lm = LinearRegression()
lm.fit(X_train, y_train)

cdf = pd.DataFrame(lm.coef_, X.columns)
print(cdf)
plt.show()