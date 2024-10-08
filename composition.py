import collections
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot

##################################################

'''
The function get_sym_dict is intended to parse a chemical formula string and return a dictionary that 
maps chemical elements to their corresponding quantities. The factor parameter is used to multiply the quantities of each element.

Here's a breakdown of the function:

Imports and Initialization:
It uses the collections module to create a defaultdict of floats, which will hold the elements and their quantities.

Regular Expression Parsing:
The function uses the re.finditer method to match all occurrences of chemical elements in the formula string f. 
The regex pattern r"([A-Z][a-z]*)\s*([-*\.\d]*)" is designed to capture element symbols (like "H", "He", etc.) and their optional amounts.

Element and Amount Extraction:
m.group(1) captures the element symbol.
m.group(2) captures the amount, which is optional. If it's present, it's converted to a float; otherwise, the amount defaults to 1.

Dictionary Population:
The function populates sym_dict with elements as keys and their scaled amounts as values (amounts are multiplied by the provided factor).

Formula Validation:
After processing all matches, if there are any characters left in the string f, it raises a CompositionError indicating an invalid formula.
'''

'''
def get_sym_dict(f, factor):
    sym_dict = collections.defaultdict(float)
    #use regex to find elements and their counts
    for m in re.finditer(r"([A-Z][a-z]*)\s*([-*\.\d]*)", f):
        el = m.group(1)  #Element symbol
        amt = 1   #Default amount
        if m.group(2).strip() != "":
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        #Replace metched part with an empty string
        f = f.replace(m.group(), "", 1)
    #Raise an error if any part of the formula is unprocessed
    if f.strip():
        raise CompositionError("{} is an invalid formula!".format(f))
    return sym_dict
'''

def get_sym_dict(f, factor):
    sym_dict = collections.defaultdict(float)
    for m in re.finditer(r"([A-Z][a-z]*)\s*([-*\.\d]*)", f):
        el = m.group(1)
        amt = 1
        if m.group(2).strip() != "":
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        f = f.replace(m.group(), "", 1)
    if f.strip():
        raise CompositionError("{} is an invalid formula!".format(f))
    return sym_dict
'''try:
    print(get_sym_dict("H2O", 1))
except CompositionError as e:
    print(e)'''

#####################################################################
'''
The parse_formula function is designed to process chemical formulas, particularly those that include parentheses 
indicating groups of elements with a multiplier outside the parentheses. 
The function replaces any instances of @ in the formula, handles nested formulas by expanding them, 
and uses the previously discussed get_sym_dict function to convert the formula into a dictionary of elements and their quantities.

Here is a detailed breakdown of the function:

Initial Replacement:
The function removes any @ characters from the input formula, although the reason for this specific character removal 
isn't clear without context.

Regex Search for Parentheses:
The function uses re.search to find the first occurrence of a group of elements within parentheses, followed by an optional multiplier.

Handling Parentheses:
If such a group is found, it determines the multiplier (factor). If no multiplier is specified, it defaults to 1.
The function then calls get_sym_dict on the contents of the parentheses with the determined factor, 
creating a dictionary that maps elements to their quantities.
It then constructs a new formula string (expanded_sym) that replaces the group and its multiplier 
with the expanded list of elements and their quantities.

Recursive Call:
After expanding the formula, the function calls itself recursively to handle any remaining nested groups of elements.

Base Case:
When there are no more parentheses in the formula, the function calls get_sym_dict with the current formula and returns the result.

def parse_formula(formula):
    """
    Args:
        formula(str) : A string formula, eg Fe2O3, Li3Fe2(PO4)3
    Returns:
        Composition with that formula
    Notes - 
        In the case of Metallofullerene formula (eg.Y3N@C80),
        the @ mark will be dropped and passed to parser.
    """
    formula = formula.replace("@", "")

    #search for nested groups of elements in paranthesis with a multiplier
    m = re.search(r"\(([^\(\)]+)\)\s*([\.\d]*)", formula)
    if m:
        factor = 1 
        if m.group(2) != "":
            factor = float(m.group(2))
        #get the symbolic dictionary from the group considering the factor
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        #expand the formula by replacing the group with individual elements
        expanded_sym = "".join(["{}{}".format(el, amt) for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        #recursively parse the expanded formula
        return parse_formula(expanded_formula)
    # no more paranthesis, parse the flat formula
    return get_sym_dict(formula, 1)
'''
def parse_formula(formula):
    """
    Args:
        formula (str): A string formula, e.g. Fe2O3, Li3Fe2(PO4)3
    Returns:
        Composition with that formula.
    Notes:
        In the case of Metallofullerene formula (e.g. Y3N@C80),
        the @ mark will be dropped and passed to parser.
    """
    # for Metallofullerene like "Y3N@C80"
    formula = formula.replace("@", "")

    m = re.search(r"\(([^\(\)]+)\)\s*([\.\d]*)", formula)
    if m:
        factor = 1
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join(["{}{}".format(el, amt)
                                for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return parse_formula(expanded_formula)
    return get_sym_dict(formula, 1)


'''try:
    print(parse_formula("H2O2(CO2)2"))
except CompositionError as e:
    print(e)'''

###############################################

class CompositionError(Exception):
    """Exception class for composition errors"""
    pass 

##########################################################################
'''
The function _fractional_composition calculates the fractional composition of each element in a chemical formula. 
Here’s a breakdown of how the function works and a suggestion to potentially improve its readability and efficiency:

Breakdown of the Function
Parsing the Formula: The function first calls parse_formula(formula) to convert the chemical formula into a dictionary (elmap) 
where keys are elements and values are their quantities.

Filtering Elements: It initializes an empty dictionary elamt to store elements whose absolute quantity is greater than or equal to 0.05. 
It also initializes natoms to count the total number of atoms (sum of absolute values).

Calculating Total Atoms: The function iterates over elmap, and for each element with a sufficient quantity, 
it adds the element and its quantity to elamt, and increments natoms.

Computing Fractional Composition: It initializes another dictionary comp_frac. For each element in elamt, it calculates its fraction of 
the total atom count and stores it in comp_frac.

Returning the Result: The function returns the dictionary comp_frac, which contains the fractional composition of each element.
'''

def _fractional_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 0.05:
            elamt[k] = v
            natoms += abs(v)
    comp_frac = {}
    for key in elamt:
        comp_frac[key] = elamt[key] / natoms
    return comp_frac

'''try:
    print(_fractional_composition("H2O"))
except CompositionError as e:
    print(e)'''

##########################################################
def _element_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    errors = []
    for k, v in elmap.items():
        if abs(v) >= 0.05:
            elamt[k] = v
            natoms += abs(v)
    return elamt

'''try:
    print(_element_composition("H2O"))
except CompositionError as e:
    print(e)'''


##################################################################
'''
The function _assign_features aims to compute and return a set of features based on a chemical formula and 
a DataFrame containing element properties. 
Here’s a detailed explanation and refinement of the function:

Explanation of the Function

Parse Formula:
Calls _fractional_composition and _element_composition to get the fractional and absolute compositions of elements in the formula.

Initialize Feature Arrays:
avg_feature and sum_feature are initialized as zero arrays with a length equal to the number of columns in elem_props.

Calculate Features:
For each element in the fractional_composition dictionary:
Updates avg_feature and sum_feature using the properties from elem_props multiplied by their compositions.
Handles exceptions if an element is not found in elem_props, prints a message, and returns an array of NaN.

Compute Additional Features:
var_feature computes the variance of properties for elements in fractional_composition.
range_feature computes the range (difference between maximum and minimum values) for these elements.

Return Features:
Returns the transposed features array.
In case of an error, returns an array of NaN.
'''
                                
'''
def _assign_features(formula, elem_props):
    try:
        fractional_composition = _fractional_composition(formula)
        element_composition = _element_composition(formula)
        avg_feature = np.zeros(len(elem_props.iloc[0]))
        sum_feature = np.zeros(len(elem_props.iloc[0]))
        for key in fractional_composition:
            try:
                avg_feature += elem_props.loc[key].values * fractional_composition[key]
                sum_feature += elem_props.loc[key].values * element_composition[key]
            except:
                print(f"The element : {key} from formula {formula} is not currently supported in our database")
                return np.array([np.nan]*len(elem_props.iloc[0])*4)
        var_feature = elem_props.loc[list(fractional_composition.keys())].var()
        range_feature = elem_props.loc[list(fractional_composition.keys())].max() - elem_props.loc[list(fractional_composition.keys())].min()
        
        features = pd.DataFrame(np.concatenate([avg_feature, sum_feature, np.array(var_feature), np.array(range_feature)]))
        features = np.concatenate([avg_feature, sum_feature, np.array(var_feature), np.array(range_feature)])
        return features.transpose()
    except:
        print(f'There was an error with the formula : {formula}. Please check the formula')
        return np.array([np.nan]*len(elem_props.iloc[0])*4)
'''
def _assign_features(formula, elem_props):
    try:
        fractional_composition = _fractional_composition(formula)
        element_composition = _element_composition(formula)
        avg_feature = np.zeros(len(elem_props.iloc[0]))
        sum_feature = np.zeros(len(elem_props.iloc[0]))
        for key in fractional_composition:
            try:
                avg_feature += elem_props.loc[key].values * fractional_composition[key]
                sum_feature += elem_props.loc[key].values * element_composition[key]
            except:
                print('The element:', key, 'from formula', formula,'is not currently supported in our database')
                return np.array([np.nan]*len(elem_props.iloc[0])*4)
        var_feature = elem_props.loc[list(fractional_composition.keys())].var()
        range_feature = elem_props.loc[list(fractional_composition.keys())].max()-elem_props.loc[list(fractional_composition.keys())].min()

        features = pd.DataFrame(np.concatenate([avg_feature, sum_feature, np.array(var_feature), np.array(range_feature)]))
        features = np.concatenate([avg_feature, sum_feature, np.array(var_feature), np.array(range_feature)])
        return features.transpose()
    except:
        print('There was an error with the formula: "'+ formula + '", please check the formatting')
        return np.array([np.nan]*len(elem_props.iloc[0])*4)


################################################################
'''def generate_features(df,reset_index = True):
    
    Parameters
    df : Pandas.DataFrame()
        Two column dataframe of form:
            df.columns.values = array['formula', 'target'], dtype = object)
    
    Return
    x : pd.DataFrame()
        Feature Matrix with NaN values filled using the median feature value for dataset
    y : pd.Series()
        Target values
    
                                   
    column_names = np.concatenate(['avg_'+elem_props.columns.values,  'sum_'+elem_props.columns.values, 
    'var_'+elem_props.columns.values, 'range_'+elem_props.columns.values])
    # empty list to store feature vectors
    features = []
    # store property values
    targets = [] 
    # store formula
    formulae = [] 

    #add values to the list using for loop
    for formula, target in zip(df['formula'], df['target']):
        features.append(_assign_features(formula, elem_props))
        targets.append(target)
        formulae.append(formula)

    #split feature vectors and target vectors as X and y 
    X = pd.DataFrame(features, columns=column_names, index = df.index.values)
    y = pd.Series(targets, index = df.index.values, name = 'target')
    formulae = pd.Series(formulae, index=df.index.values, name = 'formula')
    #Drop elements that aren't included in the elemental properties list 
    # These will be returned as feature rows completely full of Nan Values
    X.dropna(inplace=True, how = 'all')
    y = y.loc[X.index]
    formulae = formulae.loc[X.index]

    if reset_index:
        #reset dataframe indices to simplify code later
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        formulae.reset_index(drop=True, inplace=True)

    #find median value of each column
    median_values = X.median()
    #fill the missing values in each column with the column mean value
    X.fillna(median_values, inplace=True)
    return X, y, formulae
'''

def generate_features(df, reset_index=True):
    '''
    Parameters
    ----------
    df: Pandas.DataFrame()
        Two column dataframe of form: 
            df.columns.values = array(['formula', 'target'], dtype=object)

    Return
    ----------
    X: pd.DataFrame()
        Feature Matrix with NaN values filled using the median feature value for dataset
    y: pd.Series()
        Target values
    '''

    # elem_props = pd.read_excel('data/element_properties.xlsx')
    # elem_props.index = elem_props['element'].values
    # elem_props.drop(['element'], inplace=True, axis=1)
    # # print(elem_props.head())
    # # elem_props = pd.read_json('element_chem.json').T
    column_names = np.concatenate(['avg_'+elem_props.columns.values,  'sum_'+elem_props.columns.values, 'var_'+elem_props.columns.values, 'range_'+elem_props.columns.values])

    # make empty list where we will store the feature vectors
    features = []
    # make empty list where we will store the property value
    targets = []
    # store formula
    formulae = []
    # add the values to the list using a for loop
    for formula, target in zip(df['formula'], df['target']):
        features.append(_assign_features(formula, elem_props))
        targets.append(target)
        formulae.append(formula)

    # split feature vectors and targets as X and y
    X = pd.DataFrame(features, columns=column_names, index=df.index.values)
    y = pd.Series(targets, index=df.index.values, name='target')
    formulae = pd.Series(formulae, index=df.index.values, name='formula')
    # drop elements that aren't included in the elmenetal properties list.
    # These will be returned as feature rows completely full of Nan values.
    X.dropna(inplace=True, how='all')
    y = y.loc[X.index]
    formulae = formulae.loc[X.index]

    if reset_index is True:
        # reset dataframe indices to simplify code later.
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        formulae.reset_index(drop=True, inplace=True)

    # get the column names
    cols = X.columns.values
    # find the mean value of each column
    median_values = X[cols].median()
    # fill the missing values in each column with the columns mean value
    X[cols] = X[cols].fillna(median_values.iloc[0])
    return X, y, formulae

def generate_muliple_features(df, reset_index=True):
    '''
    Return
    ----------
    X: pd.DataFrame()
        Feature Matrix with NaN values filled using the median feature value for dataset
    y: pd.Series()
        Target values
    '''

    # elem_props = pd.read_excel('data/element_properties.xlsx')
    # elem_props.index = elem_props['element'].values
    # elem_props.drop(['element'], inplace=True, axis=1)
    # # print(elem_props.head())
    # # elem_props = pd.read_json('element_chem.json').T
    column_names = np.concatenate(['avg_'+elem_props.columns.values,  'sum_'+elem_props.columns.values, 'var_'+elem_props.columns.values, 'range_'+elem_props.columns.values])

    # make empty list where we will store the feature vectors
    features = []
    # make empty list where we will store the property value
    targets = []
    # store formula
    formulae = []
    # add the values to the list using a for loop
    for idx, row in df.iterrows():
        formula = row['formula']
        target_values = row.drop('formula').values
        features.append(_assign_features(formula, elem_props))
        targets.append(target_values)
        formulae.append(formula)

    # split feature vectors and targets as X and y
    X = pd.DataFrame(features, columns=column_names, index=df.index.values)
    y = pd.DataFrame(targets, index=df.index.values, columns=df.columns[1:])
    formulae = pd.Series(formulae, index=df.index.values, name='formula')
    # drop elements that aren't included in the elmenetal properties list.
    # These will be returned as feature rows completely full of Nan values.
    X.dropna(inplace=True, how='all')
    y = y.loc[X.index]
    formulae = formulae.loc[X.index]

    if reset_index is True:
        # reset dataframe indices to simplify code later.
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        formulae.reset_index(drop=True, inplace=True)

    # get the column names
    cols = X.columns.values
    # find the mean value of each column
    median_values = X[cols].median()
    # fill the missing values in each column with the columns mean value
    X[cols] = X[cols].fillna(median_values.iloc[0])
    return X, y, formulae



elem_dict = {'Atomic_Number': {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 
                               'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 
                               'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 
                               'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 
                               'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 
                               'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 
                               'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 
                               'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 
                               'Tl': 81, 'Pb': 82, 'Bi': 83, 'Th': 90, 'U': 92}, 
            'Atomic_Weight': {'H': 1.00794, 'He': 4.0026019999999995, 'Li': 6.941, 'Be': 9.01218, 'B': 10.811, 'C': 12.011, 
                              'N': 14.006739999999999, 'O': 15.9994, 'F': 18.998403, 'Ne': 20.1797, 'Na': 22.989767999999998, 
                              'Mg': 24.305, 'Al': 26.981539, 'Si': 28.0855, 'P': 30.973762, 'S': 32.066, 'Cl': 35.4527, 'Ar': 39.948, 
                              'K': 39.0983, 'Ca': 40.078, 'Sc': 44.955909999999996, 'Ti': 47.88, 'V': 50.9415, 'Cr': 51.9961, 
                              'Mn': 54.93805, 'Fe': 55.847, 'Co': 58.9332, 'Ni': 58.6934, 'Cu': 63.54600000000001, 'Zn': 65.39, 
                              'Ga': 69.723, 'Ge': 72.61, 'As': 74.92159000000001, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.8, 
                              'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585, 'Zr': 91.22399999999999, 'Nb': 92.90638, 'Mo': 95.94, 
                              'Tc': 97.9072, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 
                              'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 'I': 126.90446999999999, 'Xe': 131.29, 'Cs': 132.90543, 
                              'Ba': 137.327, 'La': 138.9055, 'Ce': 140.115, 'Pr': 140.90765, 'Nd': 144.24, 'Pm': 144.9127, 'Sm': 150.36, 
                              'Eu': 151.965, 'Gd': 157.25, 'Tb': 158.92533999999998, 'Dy': 162.5, 'Ho': 164.93032, 'Er': 167.26, 
                              'Tm': 168.93421, 'Yb': 173.04, 'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84, 'Re': 186.207, 
                              'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 'Au': 196.96653999999998, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 
                              'Bi': 208.98037, 'Th': 232.0381, 'U': 238.0289}, 
            'Period': {'H': 1, 'He': 1, 'Li': 2, 'Be': 2, 'B': 2, 'C': 2, 'N': 2, 'O': 2, 'F': 2, 'Ne': 2, 'Na': 3, 'Mg': 3, 
                       'Al': 3, 'Si': 3, 'P': 3, 'S': 3, 'Cl': 3, 'Ar': 3, 'K': 4, 'Ca': 4, 'Sc': 4, 'Ti': 4, 'V': 4, 'Cr': 4, 
                       'Mn': 4, 'Fe': 4, 'Co': 4, 'Ni': 4, 'Cu': 4, 'Zn': 4, 'Ga': 4, 'Ge': 4, 'As': 4, 'Se': 4, 'Br': 4, 'Kr': 4, 
                       'Rb': 5, 'Sr': 5, 'Y': 5, 'Zr': 5, 'Nb': 5, 'Mo': 5, 'Tc': 5, 'Ru': 5, 'Rh': 5, 'Pd': 5, 'Ag': 5, 'Cd': 5, 
                       'In': 5, 'Sn': 5, 'Sb': 5, 'Te': 5, 'I': 5, 'Xe': 5, 'Cs': 6, 'Ba': 6, 'La': 6, 'Ce': 6, 'Pr': 6, 'Nd': 6, 
                       'Pm': 6, 'Sm': 6, 'Eu': 6, 'Gd': 6, 'Tb': 6, 'Dy': 6, 'Ho': 6, 'Er': 6, 'Tm': 6, 'Yb': 6, 'Lu': 6, 'Hf': 6, 
                       'Ta': 6, 'W': 6, 'Re': 6, 'Os': 6, 'Ir': 6, 'Pt': 6, 'Au': 6, 'Hg': 6, 'Tl': 6, 'Pb': 6, 'Bi': 6, 'Th': 7, 'U': 7}, 
            'group': {'H': 1, 'He': 18, 'Li': 1, 'Be': 2, 'B': 13, 'C': 14, 'N': 15, 'O': 16, 'F': 17, 'Ne': 18, 'Na': 1, 'Mg': 2, 
                      'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6, 
                      'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 12, 'Ga': 13, 'Ge': 14, 'As': 15, 'Se': 16, 'Br': 17, 
                      'Kr': 18, 'Rb': 1, 'Sr': 2, 'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Tc': 7, 'Ru': 8, 'Rh': 9, 'Pd': 10, 'Ag': 11, 
                      'Cd': 12, 'In': 13, 'Sn': 14, 'Sb': 15, 'Te': 16, 'I': 17, 'Xe': 18, 'Cs': 1, 'Ba': 2, 'La': 3, 'Ce': 3, 'Pr': 3, 
                      'Nd': 3, 'Pm': 3, 'Sm': 3, 'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3, 'Lu': 3, 'Hf': 4, 
                      'Ta': 5, 'W': 6, 'Re': 7, 'Os': 8, 'Ir': 9, 'Pt': 10, 'Au': 11, 'Hg': 12, 'Tl': 13, 'Pb': 14, 'Bi': 15, 'Th': 3, 'U': 3},
            'families': {'H': 7, 'He': 9, 'Li': 1, 'Be': 2, 'B': 6, 'C': 7, 'N': 7, 'O': 7, 'F': 8, 'Ne': 9, 'Na': 1, 'Mg': 2, 'Al': 5, 
                         'Si': 6, 'P': 7, 'S': 7, 'Cl': 8, 'Ar': 9, 'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 4, 'Cr': 4, 'Mn': 4, 'Fe': 4, 
                         'Co': 4, 'Ni': 4, 'Cu': 4, 'Zn': 4, 'Ga': 5, 'Ge': 6, 'As': 6, 'Se': 7, 'Br': 8, 'Kr': 9, 'Rb': 1, 'Sr': 2, 
                         'Y': 3, 'Zr': 4, 'Nb': 4, 'Mo': 4, 'Tc': 4, 'Ru': 4, 'Rh': 4, 'Pd': 4, 'Ag': 4, 'Cd': 4, 'In': 5, 'Sn': 5, 
                         'Sb': 6, 'Te': 6, 'I': 8, 'Xe': 9, 'Cs': 1, 'Ba': 2, 'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Pm': 3, 'Sm': 3, 
                         'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3, 'Lu': 3, 'Hf': 4, 'Ta': 4, 'W': 4, 
                         'Re': 4, 'Os': 4, 'Ir': 4, 'Pt': 4, 'Au': 4, 'Hg': 4, 'Tl': 5, 'Pb': 5, 'Bi': 5, 'Th': 3, 'U': 3}, 
            'Metal': {'H': 0, 'He': 0, 'Li': 1, 'Be': 1, 'B': 0, 'C': 0, 'N': 0, 'O': 0, 'F': 0, 'Ne': 0, 'Na': 1, 'Mg': 1, 'Al': 1, 
                      'Si': 0, 'P': 0, 'S': 0, 'Cl': 0, 'Ar': 0, 'K': 1, 'Ca': 1, 'Sc': 1, 'Ti': 1, 'V': 1, 'Cr': 1, 'Mn': 1, 'Fe': 1, 
                      'Co': 1, 'Ni': 1, 'Cu': 1, 'Zn': 1, 'Ga': 1, 'Ge': 0, 'As': 0, 'Se': 0, 'Br': 0, 'Kr': 0, 'Rb': 1, 'Sr': 1, 'Y': 1, 
                      'Zr': 1, 'Nb': 1, 'Mo': 1, 'Tc': 1, 'Ru': 1, 'Rh': 1, 'Pd': 1, 'Ag': 1, 'Cd': 1, 'In': 1, 'Sn': 1, 'Sb': 0, 'Te': 0, 
                      'I': 0, 'Xe': 0, 'Cs': 1, 'Ba': 1, 'La': 1, 'Ce': 1, 'Pr': 1, 'Nd': 1, 'Pm': 1, 'Sm': 1, 'Eu': 1, 'Gd': 1, 'Tb': 1, 
                      'Dy': 1, 'Ho': 1, 'Er': 1, 'Tm': 1, 'Yb': 1, 'Lu': 1, 'Hf': 1, 'Ta': 1, 'W': 1, 'Re': 1, 'Os': 1, 'Ir': 1, 'Pt': 1, 
                      'Au': 1, 'Hg': 1, 'Tl': 1, 'Pb': 1, 'Bi': 1, 'Th': 1, 'U': 1}, 
            'Nonmetal': {'H': 1, 'He': 1, 'Li': 0, 'Be': 0, 'B': 0, 'C': 1, 'N': 1, 'O': 1, 'F': 1, 'Ne': 1, 'Na': 0, 'Mg': 0, 'Al': 0, 
                         'Si': 1, 'P': 1, 'S': 1, 'Cl': 1, 'Ar': 1, 'K': 0, 'Ca': 0, 'Sc': 0, 'Ti': 0, 'V': 0, 'Cr': 0, 'Mn': 0, 'Fe': 0, 
                         'Co': 0, 'Ni': 0, 'Cu': 0, 'Zn': 0, 'Ga': 0, 'Ge': 0, 'As': 0, 'Se': 1, 'Br': 1, 'Kr': 1, 'Rb': 0, 'Sr': 0, 'Y': 0, 
                         'Zr': 0, 'Nb': 0, 'Mo': 0, 'Tc': 0, 'Ru': 0, 'Rh': 0, 'Pd': 0, 'Ag': 0, 'Cd': 0, 'In': 0, 'Sn': 0, 'Sb': 1, 
                         'Te': 1, 'I': 1, 'Xe': 1, 'Cs': 0, 'Ba': 0, 'La': 0, 'Ce': 0, 'Pr': 0, 'Nd': 0, 'Pm': 0, 'Sm': 0, 'Eu': 0, 
                         'Gd': 0, 'Tb': 0, 'Dy': 0, 'Ho': 0, 'Er': 0, 'Tm': 0, 'Yb': 0, 'Lu': 0, 'Hf': 0, 'Ta': 0, 'W': 0, 'Re': 0, 
                         'Os': 0, 'Ir': 0, 'Pt': 0, 'Au': 0, 'Hg': 0, 'Tl': 0, 'Pb': 0, 'Bi': 0, 'Th': 0, 'U': 0}, 
            'Metalliod': {'H': 0, 'He': 0, 'Li': 0, 'Be': 0, 'B': 1, 'C': 0, 'N': 0, 'O': 0, 'F': 0, 'Ne': 0, 'Na': 0, 'Mg': 0, 'Al': 0, 
                          'Si': 0, 'P': 0, 'S': 0, 'Cl': 0, 'Ar': 0, 'K': 0, 'Ca': 0, 'Sc': 0, 'Ti': 0, 'V': 0, 'Cr': 0, 'Mn': 0, 'Fe': 0, 
                          'Co': 0, 'Ni': 0, 'Cu': 0, 'Zn': 0, 'Ga': 0, 'Ge': 1, 'As': 1, 'Se': 0, 'Br': 0, 'Kr': 0, 'Rb': 0, 'Sr': 0, 
                          'Y': 0, 'Zr': 0, 'Nb': 0, 'Mo': 0, 'Tc': 0, 'Ru': 0, 'Rh': 0, 'Pd': 0, 'Ag': 0, 'Cd': 0, 'In': 0, 'Sn': 0, 
                          'Sb': 0, 'Te': 0, 'I': 0, 'Xe': 0, 'Cs': 0, 'Ba': 0, 'La': 0, 'Ce': 0, 'Pr': 0, 'Nd': 0, 'Pm': 0, 'Sm': 0, 
                          'Eu': 0, 'Gd': 0, 'Tb': 0, 'Dy': 0, 'Ho': 0, 'Er': 0, 'Tm': 0, 'Yb': 0, 'Lu': 0, 'Hf': 0, 'Ta': 0, 'W': 0, 
                          'Re': 0, 'Os': 0, 'Ir': 0, 'Pt': 0, 'Au': 0, 'Hg': 0, 'Tl': 0, 'Pb': 0, 'Bi': 0, 'Th': 0, 'U': 0}, 
            'Mendeleev_Number': {'H': 92, 'He': 98, 'Li': 1, 'Be': 67, 'B': 72, 'C': 77, 'N': 82, 'O': 87, 'F': 93, 'Ne': 99, 'Na': 2, 
                                 'Mg': 68, 'Al': 73, 'Si': 78, 'P': 83, 'S': 88, 'Cl': 94, 'Ar': 100, 'K': 3, 'Ca': 7, 'Sc': 11, 
                                 'Ti': 43, 'V': 46, 'Cr': 49, 'Mn': 52, 'Fe': 55, 'Co': 58, 'Ni': 61, 'Cu': 64, 'Zn': 69, 'Ga': 74, 
                                 'Ge': 79, 'As': 84, 'Se': 89, 'Br': 95, 'Kr': 101, 'Rb': 4, 'Sr': 8, 'Y': 12, 'Zr': 44, 'Nb': 47, 
                                 'Mo': 50, 'Tc': 53, 'Ru': 56, 'Rh': 59, 'Pd': 62, 'Ag': 65, 'Cd': 70, 'In': 75, 'Sn': 80, 'Sb': 85, 
                                 'Te': 90, 'I': 96, 'Xe': 102, 'Cs': 5, 'Ba': 9, 'La': 13, 'Ce': 15, 'Pr': 17, 'Nd': 19, 'Pm': 21, 
                                 'Sm': 23, 'Eu': 25, 'Gd': 27, 'Tb': 29, 'Dy': 31, 'Ho': 33, 'Er': 35, 'Tm': 37, 'Yb': 39, 'Lu': 41, 
                                 'Hf': 45, 'Ta': 48, 'W': 51, 'Re': 54, 'Os': 57, 'Ir': 60, 'Pt': 63, 'Au': 66, 'Hg': 71, 'Tl': 76, 
                                 'Pb': 81, 'Bi': 86, 'Th': 16, 'U': 20}, 
            'l_quantum_number': {'H': 0, 'He': 0, 'Li': 0, 'Be': 0, 'B': 1, 'C': 1, 'N': 1, 'O': 1, 'F': 1, 'Ne': 1, 'Na': 0, 'Mg': 0, 
                                 'Al': 1, 'Si': 1, 'P': 1, 'S': 1, 'Cl': 1, 'Ar': 1, 'K': 0, 'Ca': 0, 'Sc': 2, 'Ti': 2, 'V': 2, 'Cr': 2, 
                                 'Mn': 2, 'Fe': 2, 'Co': 2, 'Ni': 2, 'Cu': 2, 'Zn': 2, 'Ga': 1, 'Ge': 1, 'As': 1, 'Se': 1, 'Br': 1, 
                                 'Kr': 1, 'Rb': 0, 'Sr': 0, 'Y': 2, 'Zr': 2, 'Nb': 2, 'Mo': 2, 'Tc': 2, 'Ru': 2, 'Rh': 2, 'Pd': 2, 
                                 'Ag': 0, 'Cd': 0, 'In': 1, 'Sn': 1, 'Sb': 1, 'Te': 1, 'I': 1, 'Xe': 1, 'Cs': 0, 'Ba': 0, 'La': 2, 
                                 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Pm': 3, 'Sm': 3, 'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 
                                 'Tm': 3, 'Yb': 3, 'Lu': 2, 'Hf': 2, 'Ta': 2, 'W': 2, 'Re': 2, 'Os': 2, 'Ir': 2, 'Pt': 2, 'Au': 2, 
                                 'Hg': 2, 'Tl': 1, 'Pb': 1, 'Bi': 1, 'Th': 2, 'U': 3}, 
            'Atomic_Radius': {'H': 0.53, 'He': 0.31, 'Li': 1.67, 'Be': 1.12, 'B': 0.87, 'C': 0.67, 'N': 0.56, 'O': 0.48, 'F': 0.42, 
                              'Ne': 0.38, 'Na': 1.9, 'Mg': 1.45, 'Al': 1.18, 'Si': 1.11, 'P': 0.98, 'S': 0.88, 'Cl': 0.79, 'Ar': 0.71, 
                              'K': 2.43, 'Ca': 1.94, 'Sc': 1.84, 'Ti': 1.76, 'V': 1.71, 'Cr': 1.66, 'Mn': 1.61, 'Fe': 1.56, 'Co': 1.52, 
                              'Ni': 1.49, 'Cu': 1.45, 'Zn': 1.42, 'Ga': 1.36, 'Ge': 1.25, 'As': 1.14, 'Se': 1.03, 'Br': 0.94, 'Kr': 0.88, 
                              'Rb': 2.65, 'Sr': 2.19, 'Y': 2.12, 'Zr': 2.06, 'Nb': 1.98, 'Mo': 1.9, 'Tc': 1.83, 'Ru': 1.78, 'Rh': 1.73, 
                              'Pd': 1.69, 'Ag': 1.65, 'Cd': 1.61, 'In': 1.56, 'Sn': 1.45, 'Sb': 1.33, 'Te': 1.23, 'I': 1.15, 'Xe': 1.08, 
                              'Cs': 2.98, 'Ba': 2.53, 'La': 1.95, 'Ce': 1.85, 'Pr': 2.47, 'Nd': 2.06, 'Pm': 2.05, 'Sm': 2.38, 'Eu': 2.31, 
                              'Gd': 2.33, 'Tb': 2.25, 'Dy': 2.28, 'Ho': 2.26, 'Er': 2.26, 'Tm': 2.22, 'Yb': 2.22, 'Lu': 2.17, 'Hf': 2.08, 
                              'Ta': 2.0, 'W': 1.93, 'Re': 1.88, 'Os': 1.85, 'Ir': 1.8, 'Pt': 1.77, 'Au': 1.74, 'Hg': 1.71, 'Tl': 1.56, 
                              'Pb': 1.54, 'Bi': 1.43, 'Th': 1.8, 'U': 1.75}, 
            'Miracle_Radius_[pm]': {'H': np.NaN, 'He': np.NaN, 'Li': 152.0, 'Be': 112.0, 'B': 88.0, 'C': 77.0, 'N': 72.0, 'O': 64.0, 
                                    'F': np.NaN, 'Ne': np.NaN, 'Na': 180.0, 'Mg': 160.0, 'Al': 141.0, 'Si': 110.0, 'P': 102.0, 'S': 103.0, 
                                    'Cl': np.NaN, 'Ar': np.NaN, 'K': 230.0, 'Ca': 201.0, 'Sc': 162.0, 'Ti': 142.0, 'V': 134.0, 'Cr': 130.0, 
                                    'Mn': 132.0, 'Fe': 125.0, 'Co': 125.0, 'Ni': 126.0, 'Cu': 126.0, 'Zn': 140.0, 'Ga': 134.0, 'Ge': 114.0, 
                                    'As': 115.0, 'Se': 118.0, 'Br': np.NaN, 'Kr': np.NaN, 'Rb': 244.0, 'Sr': 212.0, 'Y': 179.0, 'Zr': 158.0, 
                                    'Nb': 143.0, 'Mo': 139.0, 'Tc': 136.0, 'Ru': 134.0, 'Rh': 132.0, 'Pd': 142.0, 'Ag': 144.0, 'Cd': 157.0, 
                                    'In': 155.0, 'Sn': 155.0, 'Sb': 155.0, 'Te': 140.0, 'I': np.NaN, 'Xe': np.NaN, 'Cs': 264.0, 'Ba': 223.0, 
                                    'La': 187.0, 'Ce': 182.0, 'Pr': 183.0, 'Nd': 182.0, 'Pm': 185.0, 'Sm': 185.0, 'Eu': 196.0, 'Gd': 176.0, 
                                    'Tb': 176.0, 'Dy': 175.0, 'Ho': 177.0, 'Er': 175.0, 'Tm': 175.0, 'Yb': 190.0, 'Lu': 175.0, 'Hf': 158.0, 
                                    'Ta': 145.0, 'W': 135.0, 'Re': 137.0, 'Os': 135.0, 'Ir': 136.0, 'Pt': 139.0, 'Au': 143.0, 'Hg': 152.0, 
                                    'Tl': 172.0, 'Pb': 174.0, 'Bi': 162.0, 'Th': 178.0, 'U': 158.0}, 
            'Covalent_Radius': {'H': 0.37, 'He': 0.32, 'Li': 1.34, 'Be': 0.9, 'B': 0.82, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71, 
                                'Ne': 0.69, 'Na': 1.54, 'Mg': 1.3, 'Al': 1.18, 'Si': 1.11, 'P': 1.06, 'S': 1.02, 'Cl': 0.99, 'Ar': 0.97, 
                                'K': 1.96, 'Ca': 1.74, 'Sc': 1.44, 'Ti': 1.36, 'V': 1.25, 'Cr': 1.27, 'Mn': 1.39, 'Fe': 1.25, 'Co': 1.26, 
                                'Ni': 1.21, 'Cu': 1.38, 'Zn': 1.31, 'Ga': 1.26, 'Ge': 1.22, 'As': 1.19, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.1, 
                                'Rb': 2.11, 'Sr': 1.92, 'Y': 1.62, 'Zr': 1.48, 'Nb': 1.37, 'Mo': 1.45, 'Tc': 1.56, 'Ru': 1.26, 'Rh': 1.35, 
                                'Pd': 1.31, 'Ag': 1.53, 'Cd': 1.48, 'In': 1.44, 'Sn': 1.41, 'Sb': 1.38, 'Te': 1.35, 'I': 1.33, 'Xe': 1.3, 
                                'Cs': 2.25, 'Ba': 1.98, 'La': 1.69, 'Ce': 1.65, 'Pr': 1.65, 'Nd': 1.84, 'Pm': 1.63, 'Sm': 1.62, 'Eu': 1.85, 
                                'Gd': 1.61, 'Tb': 1.59, 'Dy': 1.59, 'Ho': 1.58, 'Er': 1.57, 'Tm': 1.56, 'Yb': 1.56, 'Lu': 1.6, 'Hf': 1.5, 
                                'Ta': 1.38, 'W': 1.46, 'Re': 1.59, 'Os': 1.28, 'Ir': 1.37, 'Pt': 1.28, 'Au': 1.44, 'Hg': 1.49, 'Tl': 1.48, 
                                'Pb': 1.47, 'Bi': 1.46, 'Th': 1.65, 'U': 1.42}, 
            'Zunger_radii_sum': {'H': 1.25, 'He': 0.0, 'Li': 1.61, 'Be': 1.08, 'B': 0.795, 'C': 0.64, 'N': 0.54, 'O': 0.465, 'F': 0.405, 
                                 'Ne': 0.0, 'Na': 2.65, 'Mg': 2.03, 'Al': 1.675, 'Si': 1.42, 'P': 1.24, 'S': 1.1, 'Cl': 1.01, 'Ar': 0.0, 
                                 'K': 3.69, 'Ca': 3.0, 'Sc': 2.75, 'Ti': 2.58, 'V': 2.43, 'Cr': 2.44, 'Mn': 2.22, 'Fe': 2.11, 'Co': 2.02, 
                                 'Ni': 2.18, 'Cu': 2.04, 'Zn': 1.88, 'Ga': 1.695, 'Ge': 1.56, 'As': 1.415, 'Se': 1.285, 'Br': 1.2, 
                                 'Kr': 0.0, 'Rb': 4.1, 'Sr': 3.21, 'Y': 2.94, 'Zr': 2.825, 'Nb': 2.76, 'Mo': 2.72, 'Tc': 2.65, 
                                 'Ru': 2.605, 'Rh': 2.52, 'Pd': 2.45, 'Ag': 2.375, 'Cd': 2.215, 'In': 2.05, 'Sn': 1.88, 'Sb': 1.765, 
                                 'Te': 1.67, 'I': 1.585, 'Xe': 0.0, 'Cs': 4.31, 'Ba': 3.4019999999999997, 'La': 3.08, 'Ce': 4.5, 
                                 'Pr': 4.48, 'Nd': 3.99, 'Pm': 3.99, 'Sm': 4.14, 'Eu': 3.94, 'Gd': 3.91, 'Tb': 3.89, 'Dy': 3.67, 
                                 'Ho': 3.65, 'Er': 3.63, 'Tm': 3.6, 'Yb': 3.59, 'Lu': 3.37, 'Hf': 2.91, 'Ta': 2.79, 'W': 2.735, 
                                 'Re': 2.68, 'Os': 2.65, 'Ir': 2.628, 'Pt': 2.7, 'Au': 2.66, 'Hg': 2.41, 'Tl': 2.235, 'Pb': 2.09, 
                                 'Bi': 1.9969999999999999, 'Th': 4.98, 'U': 4.72}, 
            'ionic_radius': {'H': 0.25, 'He': 0.31, 'Li': 1.45, 'Be': 1.05, 'B': 0.85, 'C': 0.7, 'N': 0.65, 'O': 0.6, 'F': 0.5, 
                             'Ne': 0.38, 'Na': 1.8, 'Mg': 1.5, 'Al': 1.25, 'Si': 1.1, 'P': 1.0, 'S': 1.0, 'Cl': 1.0, 'Ar': 0.71, 
                             'K': 2.2, 'Ca': 1.8, 'Sc': 1.6, 'Ti': 1.4, 'V': 1.35, 'Cr': 1.4, 'Mn': 1.4, 'Fe': 1.4, 'Co': 1.35, 
                             'Ni': 1.35,'Cu': 1.35, 'Zn': 1.35, 'Ga': 1.3, 'Ge': 1.25, 'As': 1.15, 'Se': 1.15, 'Br': 1.15, 'Kr': 0.88, 
                             'Rb': 2.35, 'Sr': 2.0, 'Y': 1.85, 'Zr': 1.55, 'Nb': 1.45, 'Mo': 1.45, 'Tc': 1.35, 'Ru': 1.3, 'Rh': 1.35, 
                             'Pd': 1.4, 'Ag': 1.6, 'Cd': 1.55, 'In': 1.55, 'Sn': 1.45, 'Sb': 1.45, 'Te': 1.4, 'I': 1.4, 'Xe': 1.08, 
                             'Cs': 2.6, 'Ba': 2.15, 'La': 1.95, 'Ce': 1.85, 'Pr': 1.85, 'Nd': 1.85, 'Pm': 1.85, 'Sm': 1.85, 'Eu': 1.85, 
                             'Gd': 1.8, 'Tb': 1.75, 'Dy': 1.75, 'Ho': 1.75, 'Er': 1.75, 'Tm': 1.75, 'Yb': 1.75, 'Lu': 1.75, 'Hf': 1.55, 
                             'Ta': 1.45, 'W': 1.35, 'Re': 1.35, 'Os': 1.3, 'Ir': 1.35, 'Pt': 1.35, 'Au': 1.35, 'Hg': 1.5, 'Tl': 1.9, 
                             'Pb': 1.8, 'Bi': 1.6, 'Th': 1.8, 'U': 1.75},
            'crystal_radius': {'H': 0.1, 'He': 0.0, 'Li': 0.9, 'Be': 0.41, 'B': 0.25, 'C': 0.29, 'N': 0.3, 'O': 1.21, 'F': 1.19, 'Ne': 0.0, 
                               'Na': 1.16, 'Mg': 0.86, 'Al': 0.53, 'Si': 0.4, 'P': 0.31, 'S': 0.43, 'Cl': 1.67, 'Ar': 0.0, 'K': 1.52, 
                               'Ca': 1.14, 'Sc': 0.89, 'Ti': 0.75, 'V': 0.68, 'Cr': 0.76, 'Mn': 0.81, 'Fe': 0.69, 'Co': 0.54, 'Ni': 0.7, 
                               'Cu': 0.71, 'Zn': 0.74, 'Ga': 0.76, 'Ge': 0.53, 'As': 0.72, 'Se': 0.56, 'Br': 1.82, 'Kr': np.NaN, 
                               'Rb': 1.66, 'Sr': 1.32, 'Y': 1.04, 'Zr': 0.86, 'Nb': 0.78, 'Mo': 0.79, 'Tc': 0.79, 'Ru': 0.82, 'Rh': 0.81, 
                               'Pd': 0.78, 'Ag': 1.29, 'Cd': 0.92, 'In': 0.94, 'Sn': 0.69, 'Sb': 0.9, 'Te': 1.11, 'I': 2.06, 'Xe': 0.62, 
                               'Cs': 1.81, 'Ba': 1.49, 'La': 1.36, 'Ce': 1.15, 'Pr': 1.32, 'Nd': 1.3, 'Pm': 1.28, 'Sm': 1.1, 'Eu': 1.31, 
                               'Gd': 1.08, 'Tb': 1.18, 'Dy': 1.05, 'Ho': 1.04, 'Er': 1.03, 'Tm': 1.02, 'Yb': 1.13, 'Lu': 1.0, 'Hf': 0.85, 
                               'Ta': 0.78, 'W': 0.74, 'Re': 0.77, 'Os': 0.77, 'Ir': 0.77, 'Pt': 0.74, 'Au': 1.51, 'Hg': 0.83, 'Tl': 1.03, 
                               'Pb': 1.49, 'Bi': 1.17, 'Th': 1.19, 'U': 0.87}, 
            'Pauling_Electronegativity': {'H': 2.2, 'He': 0.0, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 
                                          'F': 3.98, 'Ne': 0.0, 'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.9, 'P': 2.19, 'S': 2.58, 
                                          'Cl': 3.16, 'Ar': 0.0, 'K': 0.82, 'Ca': 1.0, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 
                                          'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.9, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 
                                          'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 3.0, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 
                                          'Nb': 1.6, 'Mo': 2.16, 'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.2, 'Ag': 1.93, 'Cd': 1.69, 
                                          'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 2.6, 'Cs': 0.79, 'Ba': 0.89, 
                                          'La': 1.1, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.2, 'Gd': 1.2, 
                                          'Tb': 1.2, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.1, 'Lu': 1.27, 'Hf': 1.3, 
                                          'Ta': 1.5, 'W': 2.36, 'Re': 1.9, 'Os': 2.2, 'Ir': 2.2, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.0, 
                                          'Tl': 1.62, 'Pb': 1.87, 'Bi': 2.02, 'Th': 1.3, 'U': 1.38}, 
            'MB_electonegativity': {'H': 2.1, 'He': 0.0, 'Li': 0.9, 'Be': 1.45, 'B': 1.9, 'C': 2.37, 'N': 2.85, 'O': 3.32, 'F': 3.78, 
                                    'Ne': 0.0, 'Na': 0.89, 'Mg': 1.31, 'Al': 1.64, 'Si': 1.98, 'P': 2.32, 'S': 2.65, 'Cl': 2.98, 
                                    'Ar': 0.0, 'K': 0.8, 'Ca': 1.17, 'Sc': 1.5, 'Ti': 1.86, 'V': 2.22, 'Cr': 2.0, 'Mn': 2.04, 
                                    'Fe': 1.67, 'Co': 1.72, 'Ni': 1.76, 'Cu': 1.08, 'Zn': 1.44, 'Ga': 1.7, 'Ge': 1.99, 'As': 2.27, 
                                    'Se': 2.54, 'Br': 2.83, 'Kr': np.NaN, 'Rb': 0.8, 'Sr': 1.13, 'Y': 1.41, 'Zr': 1.7, 'Nb': 2.03, 
                                    'Mo': 1.94, 'Tc': 2.18, 'Ru': 1.97, 'Rh': 1.99, 'Pd': 2.08, 'Ag': 1.07, 'Cd': 1.4, 'In': 1.63, 
                                    'Sn': 1.88, 'Sb': 2.14, 'Te': 2.38, 'I': 2.76, 'Xe': 0.0, 'Cs': 0.77, 'Ba': 1.08, 'La': 1.35, 
                                    'Ce': 1.1, 'Pr': 1.1, 'Nd': 1.2, 'Pm': 1.15, 'Sm': 1.2, 'Eu': 1.15, 'Gd': 1.1, 'Tb': 1.2, 
                                    'Dy': 1.15, 'Ho': 1.2, 'Er': 1.2, 'Tm': 1.2, 'Yb': 1.1, 'Lu': 1.2, 'Hf': 1.73, 'Ta': 1.94, 
                                    'W': 1.79, 'Re': 2.06, 'Os': 1.85, 'Ir': 1.87, 'Pt': 1.91, 'Au': 1.19, 'Hg': 1.49, 'Tl': 1.69, 
                                    'Pb': 1.92, 'Bi': 2.14, 'Th': 1.3, 'U': 1.7}, 
            'Gordy_electonegativity': {'H': 7.1784, 'He': 12.0486, 'Li': 3.2223, 'Be': 3.7942, 'B': 4.5951, 'C': 5.6246, 'N': 6.8834, 
                                       'O': 8.3703, 'F': 10.0854, 'Ne': 12.0317, 'Na': 2.5378, 'Mg': 2.9745, 'Al': 3.5237, 'Si': 4.1852, 
                                       'P': 4.9591, 'S': 5.8458, 'Cl': 6.8446, 'Ar': 7.9552, 'K': 2.7882, 'Ca': 3.0128, 'Sc': 3.0728, 
                                       'Ti': 3.1359, 'V': 3.2021, 'Cr': 3.2713, 'Mn': 3.3437, 'Fe': 3.4189999999999996, 'Co': 3.4976, 
                                       'Ni': 3.5791, 'Cu': 3.6637, 'Zn': 3.7515, 'Ga': 4.1672, 'Ge': 4.6406, 'As': 5.172000000000001, 
                                       'Se': 5.761, 'Br': 6.4079, 'Kr': 7.1127, 'Rb': 3.1886, 'Sr': 3.3588, 'Y': 3.4043, 'Zr': 3.4521, 
                                       'Nb': 3.5023, 'Mo': 3.5548, 'Tc': 3.6096, 'Ru': 3.6668, 'Rh': 3.7262, 'Pd': 3.7880000000000003, 
                                       'Ag': 3.8522, 'Cd': 3.9187, 'In': 4.2337, 'Sn': 4.5926, 'Sb': 4.9953, 'Te': 5.4418, 'I': 5.9319, 
                                       'Xe': 6.4662, 'Cs': 4.4326, 'Ba': 4.4699, 'La': 4.5168, 'Ce': 4.5734, 'Pr': 4.6395, 'Nd': 4.7152, 
                                       'Pm': 4.8006, 'Sm': 4.8955, 'Eu': 5.0001, 'Gd': 5.1142, 'Tb': 5.2379999999999995, 'Dy': 5.3714, 
                                       'Ho': 5.5143, 'Er': 5.6669, 'Tm': 5.8291, 'Yb': 6.0009, 'Lu': 6.1824, 'Hf': 6.3704, 'Ta': 6.5738, 
                                       'W': 6.7841, 'Re': 7.004, 'Os': 7.2335, 'Ir': 7.4345, 'Pt': 7.7208, 'Au': 7.9791, 'Hg': 8.2469, 
                                       'Tl': 4.6618, 'Pb': 4.7404, 'Bi': 4.8288, 'Th': 2.8784, 'U': 3.3168}, 
            'Mulliken_EN': {'H': 7.18, 'He': 0.0, 'Li': 3.01, 'Be': 4.9, 'B': 4.29, 'C': 6.27, 'N': 7.3, 'O': 7.54, 'F': 10.41, 'Ne': 0.0, 
                            'Na': 2.85, 'Mg': 3.75, 'Al': 3.23, 'Si': 4.77, 'P': 5.62, 'S': 6.22, 'Cl': 8.3, 'Ar': 0.0, 'K': 2.42, 
                            'Ca': 2.2, 'Sc': 3.34, 'Ti': 3.45, 'V': 3.6, 'Cr': 3.72, 'Mn': 3.72, 'Fe': 4.06, 'Co': 4.3, 'Ni': 4.4, 
                            'Cu': 4.48, 'Zn': 4.45, 'Ga': 3.2, 'Ge': 4.6, 'As': 5.3, 'Se': 5.89, 'Br': 7.59, 'Kr': np.NaN, 'Rb': 2.34, 
                            'Sr': 2.0, 'Y': 3.19, 'Zr': 3.64, 'Nb': 4.0, 'Mo': 3.9, 'Tc': 4.0, 'Ru': 4.5, 'Rh': 4.3, 'Pd': 4.45, 
                            'Ag': 4.44, 'Cd': 4.33, 'In': 3.1, 'Sn': 4.3, 'Sb': 4.85, 'Te': 5.49, 'I': 6.76, 'Xe': np.NaN, 'Cs': 2.18, 
                            'Ba': 2.4, 'La': 3.1, 'Ce': 3.1, 'Pr': 3.1, 'Nd': 3.1, 'Pm': 3.1, 'Sm': 3.1, 'Eu': 3.1, 'Gd': 3.1, 'Tb': 3.1, 
                            'Dy': 3.1, 'Ho': 3.1, 'Er': 3.1, 'Tm': 3.1, 'Yb': 3.1, 'Lu': 3.1, 'Hf': 3.8, 'Ta': 4.11, 'W': 4.4, 'Re': 4.02, 
                            'Os': 4.9, 'Ir': 5.4, 'Pt': 5.6, 'Au': 5.77, 'Hg': 4.91, 'Tl': 3.2, 'Pb': 3.9, 'Bi': 4.69, 'Th': 3.2, 'U': 3.4}, 
            'Allred-Rockow_electronegativity': {'H': 2.3, 'He': 4.16, 'Li': 0.912, 'Be': 1.5759999999999998, 'B': 2.051, 'C': 2.544, 
                                                'N': 3.0660000000000003, 'O': 3.61, 'F': 4.1930000000000005, 'Ne': 4.789, 
                                                'Na': 0.8690000000000001, 'Mg': 1.2930000000000001, 'Al': 1.6130000000000002, 
                                                'Si': 1.916, 'P': 2.253, 'S': 2.589, 'Cl': 2.8689999999999998, 'Ar': 3.242, 
                                                'K': 0.7340000000000001, 'Ca': 1.034, 'Sc': 1.19, 'Ti': 1.38, 'V': 1.53, 'Cr': 1.65, 
                                                'Mn': 1.75, 'Fe': 1.8, 'Co': 1.84, 'Ni': 1.88, 'Cu': 1.85, 'Zn': 1.59, 'Ga': 1.756, 
                                                'Ge': 1.994, 'As': 2.211, 'Se': 2.434, 'Br': 2.685, 'Kr': 2.966, 'Rb': 0.706, 'Sr': 0.963, 
                                                'Y': 1.12, 'Zr': 1.32, 'Nb': 1.41, 'Mo': 1.47, 'Tc': 1.51, 'Ru': 1.54, 'Rh': 1.56, 
                                                'Pd': 1.59, 'Ag': 1.87, 'Cd': 1.52, 'In': 1.656, 'Sn': 1.824, 'Sb': 1.984, 'Te': 2.158, 
                                                'I': 2.359, 'Xe': 2.582, 'Cs': 0.659, 'Ba': 0.8809999999999999, 'La': 1.09, 'Ce': 1.09, 
                                                'Pr': 1.09, 'Nd': 1.09, 'Pm': 1.09, 'Sm': 1.09, 'Eu': 1.09, 'Gd': 1.09, 'Tb': 1.09, 
                                                'Dy': 1.09, 'Ho': 1.09, 'Er': 1.09, 'Tm': 1.09, 'Yb': 1.09, 'Lu': 1.09, 'Hf': 1.16, 
                                                'Ta': 1.34, 'W': 1.47, 'Re': 1.6, 'Os': 1.65, 'Ir': 1.68, 'Pt': 1.72, 'Au': 1.92, 
                                                'Hg': 1.76, 'Tl': 1.7890000000000001, 'Pb': 1.854, 'Bi': 2.01, 'Th': 1.11, 'U': 1.22}, 
            'metallic_valence': {'H': 0.0, 'He': 0.0, 'Li': 1.0, 'Be': 2.0, 'B': 3.0, 'C': 4.0, 'N': 3.0, 'O': 0.0, 'F': 0.0, 'Ne': 0.0, 
                                 'Na': 1.0, 'Mg': 2.0, 'Al': 3.0, 'Si': 4.0, 'P': 3.0, 'S': 2.0, 'Cl': 0.0, 'Ar': 0.0, 'K': 1.0, 'Ca': 2.0, 
                                 'Sc': 3.0, 'Ti': 4.0, 'V': 5.0, 'Cr': 5.78, 'Mn': 5.78, 'Fe': 5.78, 'Co': 5.78, 'Ni': 5.78, 'Cu': 5.44, 
                                 'Zn': 4.44, 'Ga': 3.44, 'Ge': 4.0, 'As': 3.0, 'Se': 2.0, 'Br': 0.0, 'Kr': 0.0, 'Rb': 1.0, 'Sr': 2.0, 
                                 'Y': 3.0, 'Zr': 4.0, 'Nb': 5.0, 'Mo': 5.78, 'Tc': 5.78, 'Ru': 5.78, 'Rh': 5.78, 'Pd': 5.78, 'Ag': 5.44, 
                                 'Cd': 4.44, 'In': 3.44, 'Sn': 4.0, 'Sb': 3.0, 'Te': 2.0, 'I': 0.0, 'Xe': 0.0, 'Cs': 1.0, 'Ba': 2.0, 
                                 'La': 3.0, 'Ce': 3.2, 'Pr': 3.1, 'Nd': 3.1, 'Pm': 3.0, 'Sm': 2.8, 'Eu': 2.0, 'Gd': 3.0, 'Tb': 3.0, 
                                 'Dy': 3.0, 'Ho': 3.0, 'Er': 3.0, 'Tm': 3.0, 'Yb': 2.0, 'Lu': 3.0, 'Hf': 4.0, 'Ta': 5.0, 'W': 5.78, 
                                 'Re': 5.78, 'Os': 5.78, 'Ir': 5.78, 'Pt': 5.78, 'Au': 5.44, 'Hg': 4.44, 'Tl': 3.44, 'Pb': 2.44, 'Bi': 3.0, 
                                 'Th': 4.0, 'U': 5.78}, 
            'number_of_valence_electrons': {'H': 1, 'He': 0, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 0, 'Na': 1, 
                                            'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 0, 'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 
                                            'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 12, 'Ga': 3, 'Ge': 4, 
                                            'As': 5, 'Se': 6, 'Br': 7, 'Kr': 0, 'Rb': 1, 'Sr': 2, 'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 
                                            'Tc': 7, 'Ru': 8, 'Rh': 9, 'Pd': 10, 'Ag': 11, 'Cd': 12, 'In': 3, 'Sn': 4, 'Sb': 5, 'Te': 6, 
                                            'I': 7, 'Xe': 0, 'Cs': 1, 'Ba': 2, 'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Pm': 3, 'Sm': 3, 
                                            'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3, 'Lu': 3, 'Hf': 4, 
                                            'Ta': 5, 'W': 6, 'Re': 7, 'Os': 8, 'Ir': 9, 'Pt': 10, 'Au': 11, 'Hg': 12, 'Tl': 3, 'Pb': 4, 
                                            'Bi': 5, 'Th': 2, 'U': 3}, 
            'gilmor_number_of_valence_electron': {'H': 1, 'He': 0, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 0,
                                                   'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 0, 'K': 1, 'Ca': 2,
                                                   'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 4, 'Mn': 5, 'Fe': 5, 'Co': 4, 'Ni': 3, 'Cu': 2, 'Zn': 1, 
                                                   'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 0, 'Rb': 1, 'Sr': 2, 'Y': 1, 'Zr': 2, 
                                                   'Nb': 3, 'Mo': 4, 'Tc': 5, 'Ru': 5, 'Rh': 4, 'Pd': 3, 'Ag': 2, 'Cd': 1, 'In': 3, 'Sn': 4, 
                                                   'Sb': 5, 'Te': 6, 'I': 7, 'Xe': 0, 'Cs': 1, 'Ba': 2, 'La': 1, 'Ce': 3, 'Pr': 3, 'Nd': 3, 
                                                   'Pm': 3, 'Sm': 3, 'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3, 
                                                   'Lu': 3, 'Hf': 2, 'Ta': 3, 'W': 4, 'Re': 5, 'Os': 5, 'Ir': 4, 'Pt': 3, 'Au': 2, 'Hg': 1, 
                                                   'Tl': 3, 'Pb': 4, 'Bi': 5, 'Th': 2, 'U': 3}, 
            'valence_s': {'H': 1, 'He': 0, 'Li': 1, 'Be': 2, 'B': 2, 'C': 2, 'N': 2, 'O': 2, 'F': 2, 'Ne': 0, 'Na': 1, 'Mg': 2, 'Al': 2, 
                          'Si': 2, 'P': 2, 'S': 2, 'Cl': 2, 'Ar': 0, 'K': 1, 'Ca': 2, 'Sc': 2, 'Ti': 2, 'V': 2, 'Cr': 1, 'Mn': 2, 'Fe': 2, 
                          'Co': 2, 'Ni': 2, 'Cu': 1, 'Zn': 2, 'Ga': 2, 'Ge': 2, 'As': 2, 'Se': 2, 'Br': 2, 'Kr': 0, 'Rb': 1, 'Sr': 2, 'Y': 2, 
                          'Zr': 2, 'Nb': 1, 'Mo': 1, 'Tc': 2, 'Ru': 1, 'Rh': 1, 'Pd': 0, 'Ag': 1, 'Cd': 2, 'In': 2, 'Sn': 2, 'Sb': 2, 'Te': 2, 
                          'I': 2, 'Xe': 0, 'Cs': 1, 'Ba': 2, 'La': 2, 'Ce': 2, 'Pr': 2, 'Nd': 2, 'Pm': 2, 'Sm': 2, 'Eu': 2, 'Gd': 2, 'Tb': 2, 
                          'Dy': 2, 'Ho': 2, 'Er': 2, 'Tm': 2, 'Yb': 2, 'Lu': 2, 'Hf': 2, 'Ta': 2, 'W': 2, 'Re': 2, 'Os': 2, 'Ir': 2, 'Pt': 1, 
                          'Au': 1, 'Hg': 2, 'Tl': 2, 'Pb': 2, 'Bi': 2, 'Th': 2, 'U': 2}, 
            'valence_p': {'H': 0, 'He': 0, 'Li': 0, 'Be': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Ne': 0, 'Na': 0, 'Mg': 0, 'Al': 1, 
                          'Si': 2, 'P': 3, 'S': 4, 'Cl': 5, 'Ar': 0, 'K': 0, 'Ca': 0, 'Sc': 0, 'Ti': 0, 'V': 0, 'Cr': 0, 'Mn': 0, 'Fe': 0, 
                          'Co': 0, 'Ni': 0, 'Cu': 0, 'Zn': 0, 'Ga': 1, 'Ge': 2, 'As': 3, 'Se': 4, 'Br': 5, 'Kr': 0, 'Rb': 0, 'Sr': 0, 'Y': 0, 
                          'Zr': 0, 'Nb': 0, 'Mo': 0, 'Tc': 0, 'Ru': 0, 'Rh': 0, 'Pd': 0, 'Ag': 0, 'Cd': 0, 'In': 1, 'Sn': 2, 'Sb': 3, 'Te': 4, 
                          'I': 5, 'Xe': 0, 'Cs': 0, 'Ba': 0, 'La': 0, 'Ce': 0, 'Pr': 0, 'Nd': 0, 'Pm': 0, 'Sm': 0, 'Eu': 0, 'Gd': 0, 'Tb': 0, 
                          'Dy': 0, 'Ho': 0, 'Er': 0, 'Tm': 0, 'Yb': 0, 'Lu': 0, 'Hf': 0, 'Ta': 0, 'W': 0, 'Re': 0, 'Os': 0, 'Ir': 0, 'Pt': 0,
                            'Au': 0, 'Hg': 0, 'Tl': 1, 'Pb': 2, 'Bi': 3, 'Th': 0, 'U': 0}, 
            'valence_d': {'H': 0, 'He': 0, 'Li': 0, 'Be': 0, 'B': 0, 'C': 0, 'N': 0, 'O': 0, 'F': 0, 'Ne': 0, 'Na': 0, 'Mg': 0, 'Al': 0, 
                          'Si': 0, 'P': 0, 'S': 0, 'Cl': 0, 'Ar': 0, 'K': 0, 'Ca': 0, 'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 5, 'Mn': 5, 'Fe': 6, 
                          'Co': 7, 'Ni': 8, 'Cu': 10, 'Zn': 10, 'Ga': 0, 'Ge': 0, 'As': 0, 'Se': 0, 'Br': 0, 'Kr': 0, 'Rb': 0, 'Sr': 0, 
                          'Y': 1, 'Zr': 2, 'Nb': 4, 'Mo': 5, 'Tc': 5, 'Ru': 7, 'Rh': 8, 'Pd': 10, 'Ag': 10, 'Cd': 10, 'In': 10, 'Sn': 10, 
                          'Sb': 10, 'Te': 10, 'I': 10, 'Xe': 0, 'Cs': 0, 'Ba': 0, 'La': 1, 'Ce': 1, 'Pr': 0, 'Nd': 0, 'Pm': 0, 'Sm': 0, 
                          'Eu': 0, 'Gd': 1, 'Tb': 0, 'Dy': 0, 'Ho': 0, 'Er': 0, 'Tm': 0, 'Yb': 0, 'Lu': 1, 'Hf': 2, 'Ta': 3, 'W': 4, 
                          'Re': 5, 'Os': 6, 'Ir': 7, 'Pt': 9, 'Au': 10, 'Hg': 10, 'Tl': 10, 'Pb': 10, 'Bi': 10, 'Th': 2, 'U': 1}, 
            'valence_f': {'H': 0, 'He': 0, 'Li': 0, 'Be': 0, 'B': 0, 'C': 0, 'N': 0, 'O': 0, 'F': 0, 'Ne': 0, 'Na': 0, 'Mg': 0, 'Al': 0, 
                          'Si': 0, 'P': 0, 'S': 0, 'Cl': 0, 'Ar': 0, 'K': 0, 'Ca': 0, 'Sc': 0, 'Ti': 0, 'V': 0, 'Cr': 0, 'Mn': 0, 'Fe': 0, 
                          'Co': 0, 'Ni': 0, 'Cu': 0, 'Zn': 0, 'Ga': 0, 'Ge': 0, 'As': 0, 'Se': 0, 'Br': 0, 'Kr': 0, 'Rb': 0, 'Sr': 0, 'Y': 0, 
                          'Zr': 0, 'Nb': 0, 'Mo': 0, 'Tc': 0, 'Ru': 0, 'Rh': 0, 'Pd': 0, 'Ag': 0, 'Cd': 0, 'In': 0, 'Sn': 0, 'Sb': 0, 'Te': 0, 
                          'I': 0, 'Xe': 0, 'Cs': 0, 'Ba': 0, 'La': 0, 'Ce': 1, 'Pr': 3, 'Nd': 4, 'Pm': 5, 'Sm': 6, 'Eu': 7, 'Gd': 7, 'Tb': 9, 
                          'Dy': 10, 'Ho': 11, 'Er': 12, 'Tm': 13, 'Yb': 14, 'Lu': 14, 'Hf': 14, 'Ta': 14, 'W': 14, 'Re': 14, 'Os': 14, 
                          'Ir': 14, 'Pt': 14, 'Au': 14, 'Hg': 14, 'Tl': 14, 'Pb': 14, 'Bi': 14, 'Th': 0, 'U': 3}, 
            'Number_of_unfilled_s_valence_electrons': {'H': 1, 'He': 2, 'Li': 1, 'Be': 0, 'B': 0, 'C': 0, 'N': 0, 'O': 0, 'F': 0, 'Ne': 2, 
                                                       'Na': 1, 'Mg': 0, 'Al': 0, 'Si': 0, 'P': 0, 'S': 0, 'Cl': 0, 'Ar': 2, 'K': 1, 'Ca': 0, 
                                                       'Sc': 0, 'Ti': 0, 'V': 0, 'Cr': 1, 'Mn': 0, 'Fe': 0, 'Co': 0, 'Ni': 0, 'Cu': 1, 
                                                       'Zn': 0, 'Ga': 0, 'Ge': 0, 'As': 0, 'Se': 0, 'Br': 0, 'Kr': 2, 'Rb': 1, 'Sr': 0,
                                                         'Y': 0, 'Zr': 0, 'Nb': 1, 'Mo': 1, 'Tc': 0, 'Ru': 1, 'Rh': 1, 'Pd': 2, 'Ag': 1, 
                                                         'Cd': 0, 'In': 0, 'Sn': 0, 'Sb': 0, 'Te': 0, 'I': 0, 'Xe': 2, 'Cs': 1, 'Ba': 0, 
                                                         'La': 0, 'Ce': 0, 'Pr': 0, 'Nd': 0, 'Pm': 0, 'Sm': 0, 'Eu': 0, 'Gd': 0, 'Tb': 0, 
                                                         'Dy': 0, 'Ho': 0, 'Er': 0, 'Tm': 0, 'Yb': 0, 'Lu': 0, 'Hf': 0, 'Ta': 0, 'W': 0, 
                                                         'Re': 0, 'Os': 0, 'Ir': 0, 'Pt': 1, 'Au': 1, 'Hg': 0, 'Tl': 0, 'Pb': 0, 'Bi': 0, 
                                                         'Th': 0, 'U': 0}, 
            'Number_of_unfilled_p_valence_electrons': {'H': 6, 'He': 6, 'Li': 6, 'Be': 6, 'B': 5, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Ne': 6, 
                                                       'Na': 6, 'Mg': 6, 'Al': 5, 'Si': 4, 'P': 3, 'S': 2, 'Cl': 1, 'Ar': 6, 'K': 6, 'Ca': 6, 
                                                       'Sc': 6, 'Ti': 6, 'V': 6, 'Cr': 6, 'Mn': 6, 'Fe': 6, 'Co': 6, 'Ni': 6, 'Cu': 6, 
                                                       'Zn': 6, 'Ga': 5, 'Ge': 4, 'As': 3, 'Se': 2, 'Br': 1, 'Kr': 6, 'Rb': 6, 'Sr': 6, 
                                                       'Y': 6, 'Zr': 6, 'Nb': 6, 'Mo': 6, 'Tc': 6, 'Ru': 6, 'Rh': 6, 'Pd': 6, 'Ag': 6, 
                                                       'Cd': 6, 'In': 5, 'Sn': 4, 'Sb': 3, 'Te': 2, 'I': 1, 'Xe': 6, 'Cs': 6, 'Ba': 6, 
                                                       'La': 6, 'Ce': 6, 'Pr': 6, 'Nd': 6, 'Pm': 6, 'Sm': 6, 'Eu': 6, 'Gd': 6, 'Tb': 6, 
                                                       'Dy': 6, 'Ho': 6, 'Er': 6, 'Tm': 6, 'Yb': 6, 'Lu': 6, 'Hf': 6, 'Ta': 6, 'W': 6, 
                                                       'Re': 6, 'Os': 6, 'Ir': 6, 'Pt': 6, 'Au': 6, 'Hg': 6, 'Tl': 5, 'Pb': 4, 'Bi': 3, 
                                                       'Th': 6, 'U': 6}, 
            'Number_of_unfilled_d_valence_electrons': {'H': 10, 'He': 10, 'Li': 10, 'Be': 10, 'B': 10, 'C': 10, 'N': 10, 'O': 10, 'F': 10, 
                                                       'Ne': 10, 'Na': 10, 'Mg': 10, 'Al': 10, 'Si': 10, 'P': 10, 'S': 10, 'Cl': 10, 
                                                       'Ar': 10, 'K': 10, 'Ca': 10, 'Sc': 9, 'Ti': 8, 'V': 7, 'Cr': 5, 'Mn': 5, 'Fe': 4, 
                                                       'Co': 3, 'Ni': 2, 'Cu': 0, 'Zn': 0, 'Ga': 10, 'Ge': 10, 'As': 10, 'Se': 10, 
                                                       'Br': 10, 'Kr': 10, 'Rb': 10, 'Sr': 10, 'Y': 9, 'Zr': 8, 'Nb': 6, 'Mo': 5, 'Tc': 5, 
                                                       'Ru': 3, 'Rh': 2, 'Pd': 0, 'Ag': 0, 'Cd': 0, 'In': 0, 'Sn': 0, 'Sb': 0, 'Te': 0, 
                                                       'I': 0, 'Xe': 10, 'Cs': 10, 'Ba': 10, 'La': 9, 'Ce': 9, 'Pr': 10, 'Nd': 10, 
                                                       'Pm': 10, 'Sm': 10, 'Eu': 10, 'Gd': 9, 'Tb': 10, 'Dy': 10, 'Ho': 10, 'Er': 10, 
                                                       'Tm': 10, 'Yb': 10, 'Lu': 9, 'Hf': 8, 'Ta': 7, 'W': 6, 'Re': 5, 'Os': 4, 'Ir': 3, 
                                                       'Pt': 1, 'Au': 0, 'Hg': 0, 'Tl': 0, 'Pb': 0, 'Bi': 0, 'Th': 8, 'U': 9}, 
            'Number_of_unfilled_f_valence_electrons': {'H': 14, 'He': 14, 'Li': 14, 'Be': 14, 'B': 14, 'C': 14, 'N': 14, 'O': 14, 'F': 14, 
                                                       'Ne': 14, 'Na': 14, 'Mg': 14, 'Al': 14, 'Si': 14, 'P': 14, 'S': 14, 'Cl': 14, 'Ar': 14, 
                                                       'K': 14, 'Ca': 14, 'Sc': 14, 'Ti': 14, 'V': 14, 'Cr': 14, 'Mn': 14, 'Fe': 14, 'Co': 14, 
                                                       'Ni': 14, 'Cu': 14, 'Zn': 14, 'Ga': 14, 'Ge': 14, 'As': 14, 'Se': 14, 'Br': 14, 'Kr': 14, 
                                                       'Rb': 14, 'Sr': 14, 'Y': 14, 'Zr': 14, 'Nb': 14, 'Mo': 14, 'Tc': 14, 'Ru': 14, 'Rh': 14, 
                                                       'Pd': 14, 'Ag': 14, 'Cd': 14, 'In': 14, 'Sn': 14, 'Sb': 14, 'Te': 14, 'I': 14, 'Xe': 14, 
                                                       'Cs': 14, 'Ba': 14, 'La': 14, 'Ce': 13, 'Pr': 11, 'Nd': 10, 'Pm': 9, 'Sm': 8, 'Eu': 7, 
                                                       'Gd': 7, 'Tb': 5, 'Dy': 4, 'Ho': 3, 'Er': 2, 'Tm': 1, 'Yb': 0, 'Lu': 0, 'Hf': 0, 'Ta': 0, 
                                                       'W': 0, 'Re': 0, 'Os': 0, 'Ir': 0, 'Pt': 0, 'Au': 0, 'Hg': 0, 'Tl': 0, 'Pb': 0, 'Bi': 0, 
                                                       'Th': 14, 'U': 11}, 
            'outer_shell_electrons': {'H': 1, 'He': 0, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 0, 'Na': 1, 'Mg': 2, 
                                      'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 0, 'K': 1, 'Ca': 2, 'Sc': 2, 'Ti': 2, 'V': 2, 'Cr': 1, 
                                      'Mn': 2, 'Fe': 2, 'Co': 2, 'Ni': 2, 'Cu': 1, 'Zn': 2, 'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 
                                      'Kr': 0, 'Rb': 1, 'Sr': 2, 'Y': 2, 'Zr': 2, 'Nb': 1, 'Mo': 1, 'Tc': 2, 'Ru': 1, 'Rh': 1, 'Pd': 0, 
                                      'Ag': 1, 'Cd': 2, 'In': 2, 'Sn': 2, 'Sb': 2, 'Te': 2, 'I': 7, 'Xe': 0, 'Cs': 1, 'Ba': 2, 'La': 2, 
                                      'Ce': 2, 'Pr': 2, 'Nd': 2, 'Pm': 2, 'Sm': 2, 'Eu': 2, 'Gd': 2, 'Tb': 2, 'Dy': 2, 'Ho': 2, 'Er': 2, 
                                      'Tm': 2, 'Yb': 2, 'Lu': 2, 'Hf': 2, 'Ta': 2, 'W': 2, 'Re': 2, 'Os': 2, 'Ir': 2, 'Pt': 1, 'Au': 1, 
                                      'Hg': 2, 'Tl': 3, 'Pb': 4, 'Bi': 5, 'Th': 2, 'U': 2}, 
            '1st_ionization_potential_(kJ/mol)': {'H': 1312, 'He': 2372, 'Li': 520, 'Be': 899, 'B': 801, 'C': 1086, 'N': 1402, 'O': 1314, 
                                                  'F': 1681, 'Ne': 2081, 'Na': 496, 'Mg': 738, 'Al': 578, 'Si': 787, 'P': 1012, 'S': 1000, 
                                                  'Cl': 1251, 'Ar': 1521, 'K': 419, 'Ca': 590, 'Sc': 633, 'Ti': 659, 'V': 651, 'Cr': 653, 
                                                  'Mn': 717, 'Fe': 762, 'Co': 760, 'Ni': 737, 'Cu': 745, 'Zn': 906, 'Ga': 579, 'Ge': 762, 
                                                  'As': 947, 'Se': 941, 'Br': 1140, 'Kr': 1351, 'Rb': 403, 'Sr': 549, 'Y': 600, 'Zr': 640, 
                                                  'Nb': 652, 'Mo': 684, 'Tc': 702, 'Ru': 710, 'Rh': 720, 'Pd': 804, 'Ag': 731, 'Cd': 868, 
                                                  'In': 558, 'Sn': 709, 'Sb': 834, 'Te': 869, 'I': 1008, 'Xe': 1170, 'Cs': 376, 'Ba': 503, 
                                                  'La': 538, 'Ce': 534, 'Pr': 527, 'Nd': 533, 'Pm': 535, 'Sm': 545, 'Eu': 547, 'Gd': 593, 
                                                  'Tb': 569, 'Dy': 573, 'Ho': 581, 'Er': 589, 'Tm': 597, 'Yb': 603, 'Lu': 524, 'Hf': 659, 
                                                  'Ta': 761, 'W': 770, 'Re': 760, 'Os': 839, 'Ir': 878, 'Pt': 868, 'Au': 890, 'Hg': 1007, 
                                                  'Tl': 589, 'Pb': 716, 'Bi': 703, 'Th': 587, 'U': 598}, 
            'polarizability(A^3)': {'H': 0.7, 'He': 0.198, 'Li': 24.3, 'Be': 5.6, 'B': 3.0, 'C': 1.8, 'N': 1.1, 'O': 0.7929999999999999, 
                                    'F': 0.634, 'Ne': 0.396, 'Na': 23.6, 'Mg': 10.6, 'Al': 8.3, 'Si': 5.4, 'P': 3.6, 'S': 2.9, 'Cl': 2.2, 
                                    'Ar': 1.5859999999999999, 'K': 43.4, 'Ca': 22.8, 'Sc': 17.8, 'Ti': 14.6, 'V': 12.4, 'Cr': 11.6, 
                                    'Mn': 9.4, 'Fe': 8.4, 'Co': 7.5, 'Ni': 6.8, 'Cu': 6.7, 'Zn': 6.4, 'Ga': 8.1, 'Ge': 6.1, 'As': 4.3, 
                                    'Se': 3.8, 'Br': 3.1, 'Kr': 2.5, 'Rb': 47.3, 'Sr': 27.6, 'Y': 22.7, 'Zr': 17.9, 'Nb': 15.7, 'Mo': 12.8, 
                                    'Tc': 11.4, 'Ru': 9.6, 'Rh': 8.6, 'Pd': 4.8, 'Ag': 7.9, 'Cd': 7.2, 'In': 9.7, 'Sn': 7.7, 'Sb': 6.6, 
                                    'Te': 5.5, 'I': 5.0, 'Xe': 4.0, 'Cs': 59.6, 'Ba': 39.7, 'La': 31.1, 'Ce': 29.6, 'Pr': 28.2, 'Nd': 31.4, 
                                    'Pm': 30.1, 'Sm': 28.8, 'Eu': 27.7, 'Gd': 23.5, 'Tb': 25.5, 'Dy': 24.5, 'Ho': 23.6, 'Er': 22.7, 'Tm': 21.8, 
                                    'Yb': 21.0, 'Lu': 21.9, 'Hf': 16.2, 'Ta': 13.1, 'W': 11.1, 'Re': 9.7, 'Os': 8.5, 'Ir': 7.6, 'Pt': 6.5, 
                                    'Au': 6.1, 'Hg': 5.4, 'Tl': 7.6, 'Pb': 6.8, 'Bi': 7.4, 'Th': 32.1,  'U': 27.4}, 
            'Melting_point_(K)': {'H': 14.05, 'He': 0.95, 'Li': 453.65, 'Be': 1551.15, 'B': 2352.15, 'C': 3640.15, 'N': 63.25, 'O': 54.75, 
                                  'F': 53.35, 'Ne': 24.45, 'Na': 370.95, 'Mg': 922.15, 'Al': 933.15, 'Si': 1683.15, 'P': 317.25, 'S': 385.95, 
                                  'Cl': 172.15, 'Ar': 83.95, 'K': 336.4, 'Ca': 1112.15, 'Sc': 1814.15, 'Ti': 1933.15, 'V': 2163.15, 'Cr': 2130.15, 
                                  'Mn': 1517.15, 'Fe': 1808.15, 'Co': 1768.15, 'Ni': 1726.15, 'Cu': 1356.15, 'Zn': 692.75, 'Ga': 302.95,
                                    'Ge': 1220.55, 'As': 1090.15, 'Se': 490.15, 'Br': 265.95, 'Kr': 116.15, 'Rb': 312.05, 'Sr': 1042.15, 
                                    'Y': 1796.15, 'Zr': 2125.15, 'Nb': 2741.15, 'Mo': 2890.15, 'Tc': 2445.15, 'Ru': 2583.15, 'Rh': 2239.15, 
                                    'Pd': 1827.15, 'Ag': 1235.15, 'Cd': 594.05, 'In': 429.75, 'Sn': 505.15, 'Sb': 904.15, 'Te': 722.65, 
                                    'I': 386.65, 'Xe': 161.35, 'Cs': 301.55, 'Ba': 998.15, 'La': 1193.15, 'Ce': 1071.15, 'Pr': 1204.15, 
                                    'Nd': 1289.15, 'Pm': 1315.15, 'Sm': 1347.15, 'Eu': 1095.15, 'Gd': 1586.15, 'Tb': 1638.15, 'Dy': 1685.15, 
                                    'Ho': 1747.15, 'Er': 1802.15, 'Tm': 1818.15, 'Yb': 1092.15, 'Lu': 1936.15, 'Hf': 2500.15, 'Ta': 3269.15, 
                                    'W': 3683.15, 'Re': 3453.15, 'Os': 3318.15, 'Ir': 2683.15, 'Pt': 2045.15, 'Au': 1337.15, 'Hg': 234.25, 
                                    'Tl': 576.15, 'Pb': 600.65, 'Bi': 544.15, 'Th': 2023.15, 'U': 1405.15}, 
            'Boiling_Point_(K)': {'H': 20.25, 'He': 4.25, 'Li': 1615.15, 'Be': 3243.15, 'B': 4275.0, 'C': 5100.15, 'N': 77.35, 'O': 90.15, 
                                  'F': 85.05, 'Ne': 25.15, 'Na': 1156.15, 'Mg': 1363.15, 'Al': 2740.15, 'Si': 2628.15, 'P': 553.15, 'S': 717.85, 
                                  'Cl': 238.55, 'Ar': 87.45, 'K': 1033.15, 'Ca': 1757.15, 'Sc': 3105.15, 'Ti': 3560.15, 'V': 3653.15, 'Cr': 2945.15, 
                                  'Mn': 2235.15, 'Fe': 3023.15, 'Co': 3143.15, 'Ni': 3003.15, 'Cu': 2840.15, 'Zn': 1179.15, 'Ga': 2676.15, 
                                  'Ge': 3103.15, 'As': 890.15, 'Se': 958.15, 'Br': 331.95, 'Kr': 121.15, 'Rb': 959.15, 'Sr': 1657.15, 
                                  'Y': 3610.15, 'Zr': 4650.15, 'Nb': 5015.15, 'Mo': 4885.15, 'Tc': 5150.15, 'Ru': 4173.15, 'Rh': 4000.15, 
                                  'Pd': 3413.15, 'Ag': 2485.15, 'Cd': 1038.15, 'In': 2353.15, 'Sn': 2543.15, 'Sb': 2223.15, 'Te': 1262.95, 
                                  'I': 457.15, 'Xe': 166.05, 'Cs': 942.15, 'Ba': 1913.15, 'La': 3727.15, 'Ce': 3530.15, 'Pr': 3290.15, 
                                  'Nd': 3400.15, 'Pm': 3273.15, 'Sm': 2067.15, 'Eu': 1802.15, 'Gd': 3546.15, 'Tb': 3503.15, 'Dy': 2840.15, 
                                  'Ho': 2973.15, 'Er': 3141.15, 'Tm': 2223.15, 'Yb': 1469.15, 'Lu': 3675.15, 'Hf': 4873.15, 'Ta': 5698.15, 
                                  'W': 5933.15, 'Re': 5873.15, 'Os': 5303.15, 'Ir': 4403.15, 'Pt': 4100.15, 'Au': 3353.15, 'Hg': 630.15, 
                                  'Tl': 1730.15, 'Pb': 2013.15, 'Bi': 1833.15, 'Th': 5063.15, 'U': 4091.15}, 
            'Density_(g/mL)': {'H': 6.99e-05, 'He': 0.00017900000000000001, 'Li': 0.5429999999999999, 'Be': 1.85, 'B': 2.34, 'C': 2.25, 
                               'N': 0.00125, 'O': 0.00143, 'F': 0.0017, 'Ne': 0.0009, 'Na': 0.971, 'Mg': 1.74, 'Al': 2.7, 'Si': 2.33, 
                               'P': 1.82, 'S': 2.07, 'Cl': 0.00321, 'Ar': 0.0017800000000000001, 'K': 0.86, 'Ca': 1.55, 'Sc': 2.99, 
                               'Ti': 4.54, 'V': 6.11, 'Cr': 7.19, 'Mn': 7.43, 'Fe': 7.86, 'Co': 8.9, 'Ni': 8.9, 'Cu': 8.96, 'Zn': 7.13, 
                               'Ga': 5.9, 'Ge': 5.32, 'As': 5.73, 'Se': 4.79, 'Br': 3.12, 'Kr': 0.00374, 'Rb': 1.53, 'Sr': 2.54, 'Y': 4.47, 
                               'Zr': 6.51, 'Nb': 8.57, 'Mo': 10.2, 'Tc': 11.5, 'Ru': 12.4, 'Rh': 12.4, 'Pd': 12.0, 'Ag': 10.5, 'Cd': 8.65, 
                               'In': 7.31, 'Sn': 7.31, 'Sb': 6.69, 'Te': 6.24, 'I': 4.93, 'Xe': 0.00589, 'Cs': 1.87, 'Ba': 3.5, 'La': 6.15, 
                               'Ce': 6.66, 'Pr': 6.77, 'Nd': 7.0, 'Pm': 7.26, 'Sm': 7.52, 'Eu': 5.24, 'Gd': 7.9, 'Tb': 8.23, 'Dy': 8.55, 
                               'Ho': 8.8, 'Er': 9.07, 'Tm': 9.32, 'Yb': 6.97, 'Lu': 9.84, 'Hf': 13.3, 'Ta': 16.6, 'W': 19.3, 'Re': 21.0, 
                               'Os': 22.6, 'Ir': 22.4, 'Pt': 21.4, 'Au': 19.3, 'Hg': 13.5, 'Tl': 11.9, 'Pb': 11.4, 'Bi': 9.75, 'Th': 11.7, 'U': 19.0}, 
            'specific_heat_(J/g_K)_': {'H': 14.304, 'He': 5.193, 'Li': 3.6, 'Be': 1.82, 'B': 1.02, 'C': 0.71, 'N': 1.04, 'O': 0.92, 'F': 0.82, 
                                       'Ne': 0.904, 'Na': 1.23, 'Mg': 1.02, 'Al': 0.9, 'Si': 0.71, 'P': 0.77, 'S': 0.71, 'Cl': 0.48, 'Ar': 0.52, 
                                       'K': 0.75, 'Ca': 0.63, 'Sc': 0.6, 'Ti': 0.52, 'V': 0.49, 'Cr': 0.45, 'Mn': 0.48, 'Fe': 0.44, 'Co': 0.42, 
                                       'Ni': 0.44, 'Cu': 0.38, 'Zn': 0.39, 'Ga': 0.37, 'Ge': 0.32, 'As': 0.33, 'Se': 0.32, 'Br': 0.473, 'Kr': 0.248, 
                                       'Rb': 0.363, 'Sr': 0.3, 'Y': 0.3, 'Zr': 0.27, 'Nb': 0.26, 'Mo': 0.25, 'Tc': 0.21, 'Ru': 0.23800000000000002, 
                                       'Rh': 0.242, 'Pd': 0.24, 'Ag': 0.235, 'Cd': 0.23, 'In': 0.23, 'Sn': 0.22699999999999998, 'Sb': 0.21, 
                                       'Te': 0.2, 'I': 0.214, 'Xe': 0.158, 'Cs': 0.24, 'Ba': 0.204, 'La': 0.19, 'Ce': 0.19, 'Pr': 0.19, 'Nd': 0.19, 
                                       'Pm': 0.18, 'Sm': 0.2, 'Eu': 0.18, 'Gd': 0.23, 'Tb': 0.18, 'Dy': 0.17, 'Ho': 0.16, 'Er': 0.17, 'Tm': 0.16, 
                                       'Yb': 0.15, 'Lu': 0.15, 'Hf': 0.14, 'Ta': 0.14, 'W': 0.13, 'Re': 0.13, 'Os': 0.13, 'Ir': 0.13, 'Pt': 0.13, 
                                       'Au': 0.128, 'Hg': 0.139, 'Tl': 0.13, 'Pb': 0.13, 'Bi': 0.12, 'Th': 0.12, 'U': 0.12}, 
            'heat_of_fusion_(kJ/mol)_': {'H': 0.5868, 'He': 0.02, 'Li': 3.0, 'Be': 7.95, 'B': 50.2, 'C': 105.0, 'N': 0.3604, 'O': 0.22259, 
                                         'F': 0.2552, 'Ne': 0.3317, 'Na': 2.5980000000000003, 'Mg': 8.954, 'Al': 10.79, 'Si': 50.55, 'P': 0.657, 
                                         'S': 1.7175, 'Cl': 3.23, 'Ar': 1.188, 'K': 2.334, 'Ca': 8.54, 'Sc': 14.1, 'Ti': 15.45, 'V': 20.9, 
                                         'Cr': 16.9, 'Mn': 12.05, 'Fe': 13.8, 'Co': 16.19, 'Ni': 17.47, 'Cu': 13.05, 'Zn': 7.322, 'Ga': 5.59, 
                                         'Ge': 36.94, 'As': 27.7, 'Se': 6.694, 'Br': 5.2860000000000005, 'Kr': 1.6380000000000001, 
                                         'Rb': 2.1919999999999997, 'Sr': 8.3, 'Y': 11.4, 'Zr': 16.9, 'Nb': 26.4, 'Mo': 32.0, 'Tc': 24.0, 
                                         'Ru': 24.0, 'Rh': 21.5, 'Pd': 17.6, 'Ag': 11.3, 'Cd': 6.192, 'In': 3.263, 'Sn': 7.029, 'Sb': 19.87, 
                                         'Te': 17.49, 'I': 7.824, 'Xe': 2.2969999999999997, 'Cs': 2.092, 'Ba': 7.75, 'La': 6.2, 'Ce': 5.46, 
                                         'Pr': 6.89, 'Nd': 7.14, 'Pm': 7.88, 'Sm': 8.63, 'Eu': 9.21, 'Gd': 10.05, 'Tb': 10.8, 'Dy': 11.06, 
                                         'Ho': 12.2, 'Er': 19.9, 'Tm': 16.84, 'Yb': 7.66, 'Lu': 18.6, 'Hf': 24.06, 'Ta': 31.6, 'W': 35.4, 
                                         'Re': 33.2, 'Os': 31.8, 'Ir': 26.1, 'Pt': 19.6, 'Au': 12.55, 'Hg': 2.295, 'Tl': 4.1419999999999995, 
                                         'Pb': 4.7989999999999995, 'Bi': 11.3, 'Th': 16.1, 'U': 14.0}, 
            'heat_of_vaporization_(kJ/mol)_': {'H': 0.44936000000000004, 'He': 0.0845, 'Li': 145.92, 'Be': 292.4, 'B': 489.7, 'C': 355.8, 
                                               'N': 2.7928, 'O': 3.4099, 'F': 3.2698, 'Ne': 1.7326, 'Na': 96.96, 'Mg': 127.4, 'Al': 293.4, 
                                               'Si': 384.22, 'P': 12.129000000000001, 'S': 9.8, 'Cl': 10.2, 'Ar': 6.447, 'K': 79.87, 
                                               'Ca': 153.3, 'Sc': 314.2, 'Ti': 421.0, 'V': 453.0, 'Cr': 344.3, 'Mn': 226.0, 'Fe': 349.6, 
                                               'Co': 376.5, 'Ni': 370.4, 'Cu': 300.3, 'Zn': 115.3, 'Ga': 258.7, 'Ge': 330.9, 'As': 34.76, 
                                               'Se': 37.7, 'Br': 15.437999999999999, 'Kr': 9.029, 'Rb': 72.21600000000001, 'Sr': 144.0, 
                                               'Y': 363.0, 'Zr': 582.0, 'Nb': 682.0, 'Mo': 598.0, 'Tc': 660.0, 'Ru': 595.0, 'Rh': 493.0, 
                                               'Pd': 357.0, 'Ag': 250.58, 'Cd': 99.57, 'In': 231.5, 'Sn': 295.8, 'Sb': 77.14, 'Te': 52.55, 
                                               'I': 20.752, 'Xe': 12.636, 'Cs': 67.74, 'Ba': 142.0, 'La': 414.0, 'Ce': 414.0, 'Pr': 296.8, 
                                               'Nd': 273.0, 'Pm': 214.0, 'Sm': 166.4, 'Eu': 143.5, 'Gd': 359.4, 'Tb': 330.9, 'Dy': 230.1, 
                                               'Ho': 241.0, 'Er': 261.0, 'Tm': 191.0, 'Yb': 128.9, 'Lu': 355.9, 'Hf': 575.0, 'Ta': 743.0, 
                                               'W': 824.0, 'Re': 715.0, 'Os': 746.0, 'Ir': 604.0, 'Pt': 510.0, 'Au': 334.4, 'Hg': 59.229, 
                                               'Tl': 164.1, 'Pb': 177.7, 'Bi': 104.8, 'Th': 514.4, 'U': 477.0}, 
            'thermal_conductivity_(W/(m_K))_': {'H': 0.1815, 'He': 0.152, 'Li': 84.7, 'Be': 200.0, 'B': 27.0, 'C': 129.0, 'N': 0.02598, 
                                                'O': 0.026739999999999996, 'F': 0.0279, 'Ne': 0.0493, 'Na': 141.0, 'Mg': 156.0, 'Al': 237.0, 
                                                'Si': 148.0, 'P': 0.235, 'S': 0.26899999999999996, 'Cl': 0.0089, 'Ar': 0.01772, 'K': 102.4, 
                                                'Ca': 200.0, 'Sc': 15.8, 'Ti': 21.9, 'V': 30.7, 'Cr': 93.7, 'Mn': 7.82, 'Fe': 80.2, 'Co': 100.0, 
                                                'Ni': 90.7, 'Cu': 401.0, 'Zn': 116.0, 'Ga': 40.6, 'Ge': 59.9, 'As': 50.0, 'Se': 0.52, 'Br': 0.122, 
                                                'Kr': 0.00949, 'Rb': 58.2, 'Sr': 35.3, 'Y': 17.2, 'Zr': 22.7, 'Nb': 53.7, 'Mo': 138.0, 'Tc': 50.6, 
                                                'Ru': 117.0, 'Rh': 150.0, 'Pd': 71.8, 'Ag': 429.0, 'Cd': 96.8, 'In': 81.6, 'Sn': 66.6, 'Sb': 24.3, 
                                                'Te': 2.35, 'I': 0.449, 'Xe': 0.00569, 'Cs': 35.9, 'Ba': 18.4, 'La': 13.5, 'Ce': 11.4, 'Pr': 12.5, 
                                                'Nd': 16.5, 'Pm': 15.0, 'Sm': 13.3, 'Eu': 13.9, 'Gd': 10.6, 'Tb': 11.1, 'Dy': 10.7, 'Ho': 16.2, 
                                                'Er': 14.3, 'Tm': 16.8, 'Yb': 34.9, 'Lu': 16.4, 'Hf': 23.0, 'Ta': 57.5, 'W': 174.0, 'Re': 47.9, 
                                                'Os': 87.6, 'Ir': 147.0, 'Pt': 71.6, 'Au': 317.0, 'Hg': 8.34, 'Tl': 46.1, 'Pb': 35.3, 'Bi': 7.87, 
                                                'Th': 54.0, 'U': 27.6}, 
            'heat_atomization(kJ/mol)': {'H': 218, 'He': 0, 'Li': 161, 'Be': 324, 'B': 573, 'C': 717, 'N': 473, 'O': 249, 'F': 79, 'Ne': 0, 
                                         'Na': 109, 'Mg': 148, 'Al': 326, 'Si': 452, 'P': 315, 'S': 279, 'Cl': 121, 'Ar': 0, 'K': 90, 'Ca': 178,
                                           'Sc': 378, 'Ti': 470, 'V': 514, 'Cr': 397, 'Mn': 281, 'Fe': 418, 'Co': 425, 'Ni': 430, 'Cu': 338, 
                                           'Zn': 131, 'Ga': 286, 'Ge': 377, 'As': 302, 'Se': 227, 'Br': 112, 'Kr': 0, 'Rb': 86, 'Sr': 164,
                                             'Y': 423, 'Zr': 609, 'Nb': 726, 'Mo': 658, 'Tc': 677, 'Ru': 643, 'Rh': 556, 'Pd': 378, 'Ag': 284, 
                                             'Cd': 112, 'In': 243, 'Sn': 302, 'Sb': 262, 'Te': 197, 'I': 107, 'Xe': 0, 'Cs': 79, 'Ba': 180, 
                                             'La': 423, 'Ce': 419, 'Pr': 356, 'Nd': 328, 'Pm': 301, 'Sm': 207, 'Eu': 178, 'Gd': 398, 'Tb': 389, 
                                             'Dy': 291, 'Ho': 301, 'Er': 317, 'Tm': 232, 'Yb': 152, 'Lu': 427, 'Hf': 619, 'Ta': 782, 'W': 849,
                                         'Re': 770, 'Os': 791, 'Ir': 665, 'Pt': 565, 'Au': 366, 'Hg': 61, 'Tl': 182, 'Pb': 196, 'Bi': 207, 
                                         'Th': 576, 'U': 490}, 
            'Cohesive_energy': {'H': 0.0, 'He': 0.0, 'Li': 1.63, 'Be': 3.32, 'B': 5.81,'C': 7.37, 'N': 4.92, 'O': 2.62, 'F': 0.84, 'Ne': 0.02, 
                                'Na': 1.113, 'Mg': 1.51, 'Al': 3.39, 'Si': 4.63, 'P': 3.43, 'S': 2.85, 'Cl': 1.4, 'Ar': 0.08, 'K': 0.934, 
                                'Ca': 1.84, 'Sc': 3.9, 'Ti': 4.85, 'V': 5.31, 'Cr': 4.1, 'Mn': 2.92, 'Fe': 4.28, 'Co': 4.39, 'Ni': 4.44, 
                                'Cu': 3.49, 'Zn': 1.35, 'Ga': 2.81, 'Ge': 3.85, 'As': 2.96, 'Se': 2.46, 'Br': 1.22, 'Kr': 0.11599999999999999, 
                                'Rb': 0.852, 'Sr': 1.72, 'Y': 4.37, 'Zr': 6.25, 'Nb': 7.57, 'Mo': 6.82, 'Tc': 6.85, 'Ru': 6.74, 'Rh': 5.75, 
                                'Pd': 3.89, 'Ag': 2.95, 'Cd': 1.16, 'In': 2.52, 'Sn': 3.14, 'Sb': 2.75, 'Te': 2.19, 'I': 1.11, 'Xe': 0.16, 
                                'Cs': 0.804, 'Ba': 1.9, 'La': 4.47, 'Ce': 4.32, 'Pr': 3.7, 'Nd': 3.4, 'Pm': 0.0, 'Sm': 2.14, 'Eu': 1.86, 'Gd': 4.14, 
                                'Tb': 4.05, 'Dy': 3.04, 'Ho': 3.14, 'Er': 3.29, 'Tm': 2.42, 'Yb': 1.6, 'Lu': 4.43, 'Hf': 6.44, 'Ta': 8.1, 
                                'W': 8.9, 'Re': 8.03, 'Os': 8.17, 'Ir': 6.94, 'Pt': 5.84, 'Au': 3.81, 'Hg': 0.67, 'Tl': 1.88, 'Pb': 2.03, 
                                'Bi': 2.18, 'Th': 6.2, 'U': 5.55}}

elem_props = pd.DataFrame(elem_dict)