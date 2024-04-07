import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
 
# Introduce an error by providing an incorrect file path
df = pd.read_csv("data/train_wrong.csv")  # incorrect file path
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()
 
# Attempt to fit the model using incorrect data
model = LogisticRegression().fit(X, y)
 
with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
