import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

try:
    # incorrect file path
    df = pd.read_csv("data/trainsbeq.csv")  
    X = df.drop(columns=['Disease']).to_numpy()
    y = df['Disease'].to_numpy()
    labels = np.sort(np.unique(y))
    y = np.array([np.where(labels == x) for x in y]).flatten()

    # Attempt to fit the model using incorrect data
    model = RandomForestClassifier().fit(X, y)

    with open("model.pkl", 'wb') as f:
        pickle.dump(model, f)

except FileNotFoundError:
    print("Error: The file 'trainsbeq.csv' does not exist.")
