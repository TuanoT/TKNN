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

c = Classifier()
c.fit(X, y, k=5)