import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer.keys()
df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

