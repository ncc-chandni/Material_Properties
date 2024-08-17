Here's a polished version of your README file:

---

# Materials Property Prediction Models

This repository contains models designed to predict material properties based on their composition. 

## Files Overview

- **Data/materials.csv**: Contains 8 columns and 2382 rows. This dataset includes four material properties:
  - Bulk Modulus
  - Poisson Ratio
  - Shear Modulus
  - Elastic Anisotropy
  
- **Properties**: A detailed explanation of the properties included in the dataset.

- **composition.py**: Contains functions for processing chemical formulas and generating feature vectors based on elemental properties.

- **composition.txt**: Provides an explanation of each function in `composition.py`.

- **utils.py**: A toolkit for evaluating machine learning models, visualizing grid search results, normalizing data for effective visualization, and more.

- **utils.txt**: Describes the functions included in `utils.py`.

- **material_properties_SVM.py**: Implements a machine learning model using Support Vector Machines (SVM) to predict the Bulk Modulus of materials. This script demonstrates the end-to-end process from data cleaning to modeling and prediction, and evaluates model performance for predicting Bulk Modulus of new elements. This model is further explored in other files.

- **model_comparison.ipynb**: Compares various machine learning models—Linear Regression (LR), K-Nearest Neighbors (KNN), Decision Trees (DT), Support Vector Machines (SVM), Random Forest Regressor (RFR), and Gradient Boosting Regressor (GBR)—for predicting Bulk Modulus. The SVM model is found to perform the best among them.

- **all_properties_SVM.ipynb**: Extends the SVM model to predict all properties (Bulk Modulus, Poisson Ratio, Shear Modulus, and Elastic Anisotropy) based on the findings from the model comparison. It was discovered that SVM performs well only for Bulk Modulus and Shear Modulus, with recommendations provided for addressing this limitation.

