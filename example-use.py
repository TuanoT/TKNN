import pandas as pd
import ast

from tknn import Classifier

# Load the dataset from a CSV file
df = pd.read_csv('data/Fluorescence_olive_oil_dataset.csv')

# Keep only rows where Led is 1 and Repetition is 1
df = df[(df['Led'] == 1) & (df['Repetition'] == 1)]

# Convert the 'Data' column to X
df['Data'] = df['Data'].apply(ast.literal_eval)

X = df['Data'].tolist()
y = df['Quality'].tolist()

X_testing = X.pop(0)
X_training = X

y_testing = y.pop(0)
y_training = y

c = Classifier()
c.fit(X_training, y_training, k=5)

print(c.predict(X_testing))
print(y_testing)