# =============================================================================
#                               Import Libraries
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import composition
import utils

# =============================================================================
#                               Clean the data
# =============================================================================
df = pd.read_csv("/Users/umakantmanore/Desktop/amu/Dev_Enviroment2023/test_env/Material_Properties/materials.csv")

uncleaned_formulae = df['ENTRY ']
cleaned_formulae = []

for cell_value in uncleaned_formulae:
    split_list = cell_value.split(" [")
    clean_formula = split_list[0]
    cleaned_formulae.append(clean_formula)

df_cleaned = pd.DataFrame()
df_cleaned['formula'] = cleaned_formulae

# Define all target properties you want to predict
target_properties = ['AEL VRH bulk modulus ', 'AEL elastic anisotropy ',
                     'AEL Poisson ratio ', 'AEL VRH shear modulus ']

for prop in target_properties:
    df_cleaned[prop] = df[prop]

# =============================================================================
#                             Featurize the data
# =============================================================================
df_cleaned.columns = ['formula'] + target_properties
X, y, formulae = composition.generate_features(df_cleaned)

# =============================================================================
#                           Make a train-test split
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# =============================================================================
#                          Consider Scaling the data
# =============================================================================
scalar = StandardScaler()
normalizer = Normalizer()

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)
X_train_scaled = normalizer.fit_transform(X_train_scaled)
X_test_scaled = normalizer.transform(X_test_scaled)

# =============================================================================
#                          Select desired algorithm
# =============================================================================
base_model = SVR()

# =============================================================================
#                          Optimize Parameters
# =============================================================================
cv = KFold(n_splits=5, shuffle=True, random_state=1)
c_parameters = np.logspace(-1, 3, 5)
gamma_parameters = np.logspace(-2, 2, 5)

parameter_candidates = {'C': c_parameters, 'gamma': gamma_parameters}

grid = GridSearchCV(estimator=base_model, param_grid=parameter_candidates, cv=cv)
grid.fit(X_train_scaled, y_train)

best_parameters = grid.best_params_
print(best_parameters)

utils.plot_2d_grid_search(grid, midpoint=0.7, vmin=-0, vmax=1)
plt.plot()

# =============================================================================
#                     Check performance on the test set
# =============================================================================
final_model = MultiOutputRegressor(SVR(**best_parameters))
final_model.fit(X_train_scaled, y_train)

y_test_predicted = final_model.predict(X_test_scaled)

# Plotting actual vs predicted for each property
for i, prop in enumerate(target_properties):
    utils.plot_act_vs_pred(y_test[:, i], y_test_predicted[:, i], title=prop)
    score = r2_score(y_test[:, i], y_test_predicted[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_test_predicted[:, i]))
    print(f'{prop} - r2 score: {score:.3f}, rmse: {rmse:.2f}')

# =============================================================================
#                     Make predictions on new compounds
# =============================================================================
class MaterialsModel():
    def __init__(self, trained_model, scalar, normalizer, properties):
        self.model = trained_model
        self.scalar = scalar
        self.normalizer = normalizer
        self.properties = properties

    def predict(self, formula):
        if isinstance(formula, str):
            df_formula = pd.DataFrame({'formula': [formula], 'target': [0]*len(self.properties)})
        elif isinstance(formula, list):
            df_formula = pd.DataFrame({'formula': formula, 'target': np.zeros(len(formula))})

        X, _, formula = composition.generate_features(df_formula)
        X_scaled = self.scalar.transform(X)
        X_scaled = self.normalizer.transform(X_scaled)
        y_predicted = self.model.predict(X_scaled)

        prediction = pd.DataFrame(formula)
        for i, prop in enumerate(self.properties):
            prediction[prop] = y_predicted[:, i]
        return prediction

bulk_modulus_model = MaterialsModel(final_model, scalar, normalizer, target_properties)

formulae_to_predict = ['NaCl', 'Pu2O4', 'NaNO3']
bulk_modulus_prediction = bulk_modulus_model.predict(formulae_to_predict)
print(bulk_modulus_prediction)
