import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np


df = pd.read_csv("dataaaaqa/train.csv") 
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

#  error during testing
try:
    # Fit the model using the correct data
    model = LogisticRegression().fit(X, y)

    with open("model.pkl", 'wb') as f:
        pickle.dump(model, f)

   
    #try to load a non-existent file
    with open("non_existent_file.pkl", 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: The file does not exist.")
