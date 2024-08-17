import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import r2_score, mean_squared_error 
import composition
import utils

df = pd.read_csv("/Users/umakantmanore/Desktop/amu/Dev_Enviroment2023/test_env/Material_Properties/materials.csv")
uncleaned_formulae = df['ENTRY ']
cleaned_formulae = []

for value in uncleaned_formulae:
    #split string into list
    split_list = value.split(" [")
    clean_formula = split_list[0]
    cleaned_formulae.append(clean_formula)

#Lets now create a new dataframe to hold the clean data 
df_cleaned = pd.DataFrame()

#adding columns to the DF
df_cleaned['formula'] = cleaned_formulae

#adding a column with the target property we want to predict
#Here I am targeting to predict bulk modulus of the material
df_cleaned['bulk_modulus'] = df['AEL VRH bulk modulus ']

check_for_duplicates = df_cleaned['formula'].value_counts()

df_cleaned.drop_duplicates('formula', keep='first', inplace=True)

plt.figure(figsize=(6,4))
df_cleaned['bulk_modulus'].hist(bins=20, grid=False, edgecolor = 'black')
plt.show()

#Rename columns to match the required input
df_cleaned.columns = ['formula', 'target']

#Lets convert our chemical formula into features
X, y, formulae = composition.generate_features(df_cleaned)