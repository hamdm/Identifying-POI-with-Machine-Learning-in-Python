import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tester import dump_classifier_and_data, test_classifier
import warnings
warnings.filterwarnings('ignore')
from feature_format import featureFormat, targetFeatureSplit


pd.set_option('display.max_columns', None)
file = 'fraud_dataset.pkl'

'''
Loading the dictionary containing dataset in data_dict
& in Pandas DataFrame
'''

with open(file, "rb") as data_file:
    data_dict = pickle.load(data_file)
    
enron = pd.DataFrame(pd.read_pickle(file)).transpose()
enron.head(5)


'''
Utility Functions
'''

### Utility Functions to Analyse Data Loaded in Pandas Data Frame
# * The following utility functions analyse the enron data:
# * 1) To identify total features
# * 2) To identify persons of interest
# * 3) To replace 'NaN' with np.nan 
# * 4) To convert the null values in the financial features to zero 
# * 5) To impute the null values in the email features with the mean of the column
# * 6) Functionn to dump the classifier

def transform_null(dataframe,financialFeatures,emailFeatures):
    '''
    This function transforms the null values in the financial data to zero, 
    and imputes the null values in the email features with the mean of the column
    '''

    for feature in dataframe.columns:
        if feature in financialFeatures:
            dataframe[feature] = dataframe[feature].fillna(0)
        elif feature in emailFeatures:
            mean = dataframe[feature].mean()
            dataframe[feature] = dataframe[feature].fillna(mean)
    
    return dataframe
    
    
def convert_to_NaNs(dataframe):
    '''
    The function converts all the occurrences of NaNs to np.nan
    '''
    for col in dataframe.columns:
        dataframe[col] = dataframe[col].apply(lambda x: np.nan if str(x).strip()=='NaN' else x)
    
    nullByRow,nullByColumn = (nullValues(dataframe))
    
    return dataframe,nullByRow,nullByColumn
   
def nullValues(dataframe):
    '''
    The function counts null values by rows and columns
    '''
    nullByRow = dataframe.isnull().sum(axis=1)
    nullByColumn = dataframe.isnull().sum(axis=0)
    
    
    return nullByRow,nullByColumn
 
def inspect_data(dataframe):
    '''
    The function...
    '''
    
    print('There are {} features, and {} persons in the enron data \n'.format(len(dataframe.columns),len(dataframe.index)))
    
    poi_npoi = dataframe['poi'].value_counts().to_dict()
    print('Of the {} persons, {} are classified as Persons of Interest \n'.format(len(dataframe.index),poi_npoi[True]))
    
    dataframe, nullrows,nullcolumns =convert_to_NaNs(enron)
    print('The following columns: \n {} have null values greater than 100 \n'.format(nullcolumns[nullcolumns>100]))
    print('The following rows: \n {} \n have null values greater than 15 \n'.format(nullrows[nullrows>15]))

    return dataframe

def evaluate_model(grid, X, y, cv):
    nested_score = cross_val_score(grid, X=X, y=y, cv=cv, n_jobs=-1)
    print("Nested f1 score: {}".format(nested_score.mean()))

    grid.fit(X, y)    
    print("Best parameters: {}".format(grid.best_params_))

    cv_accuracy = []
    cv_precision = []
    cv_recall = []
    cv_f1 = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid.best_estimator_.fit(X_train, y_train)
        pred = grid.best_estimator_.predict(X_test)

        cv_accuracy.append(accuracy_score(y_test, pred))
        cv_precision.append(precision_score(y_test, pred))
        cv_recall.append(recall_score(y_test, pred))
        cv_f1.append(f1_score(y_test, pred))

    print("Mean Accuracy: {}".format(np.mean(cv_accuracy)))
    print ("Mean Precision: {}".format(np.mean(cv_precision)))
    print ("Mean Recall: {}".format(np.mean(cv_recall)))
    print ("Mean f1: {}".format(np.mean(cv_f1)))
    
def my_dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "wb") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "wb") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "wb") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

'''
Analyzing Data:
'''
data = transform_null(inspect_data(enron),financialFeatures,emailFeatures)

### Output of Analysis From The Utility Functions Used
# * Features: 21 
# * Persons: 146
# * Persons of Interest: 18
# * A record that doesn't go well with other records: THE TRAVEL AGENCY IN THE PARK 
# * A record that has null values for all features: LOCKHART EUGENE E   
    

'''
Outlier Investigation, and dropping inconsistent rows & columns
'''


data_ML = data.drop('email_address',axis=1)
data_ML.plot.scatter('salary','bonus')
data_ML.drop(['LOCKHART EUGENE E','THE TRAVEL AGENCY IN THE PARK','TOTAL'],inplace=True)
print('Printing INDEX')
print(data_ML.index)
data_ML['salary'].idxmax()

# * The analysis reveals that there is a record 'Lockhart Eugene E' which has null values for all the columns, and we shall drop that record. 
# * Similarly, there is a very weird record by the name THE TRAVEL AGENCY IN THE PARK, and that shall be dropped too. 
# * Also, the email_address column is a string and isn't really needed in the machine learning, and shall be dropped as well
# * In a scatter plot between SALARY and BONUS, there is a salary observed which is greater than 2.5 *10^7 (way huge salary for enron). Using enron['salary'].idxmax() we observed that the index for this outlier is 'Total'

'''
After droping the outliers
'''

data_ML.plot.scatter('salary','bonus')


enronData = data_ML
enronData


'''
Creating New Features
'''

enronData['fraction_from_poi'] = enronData['from_poi_to_this_person'] / enronData['to_messages']
enronData['fraction_to_poi'] = enronData['from_this_person_to_poi'] / enronData['from_messages']

ax = enronData[enronData['poi'] == False].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', color='blue', label='non-poi')
enronData[enronData['poi'] == True].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', color='red', label='poi', ax=ax)


'''
Classifiers, and parameters tuning
'''

# For model building, I will use the following strategies:
# 1) Scaling the features, and for that purpose I will use the MinMaxScaler from the
#    sklearn. Many machine learning algorithms are expected to perform better with scaled data.
# 2) Tuning the algorithms by adjusting parameters in order to maximise the evaluation metrics,
#     and for that purpose I will use the GridSearchCV from sklearn. Among the possibilities,
#     GridSearchCV exhaustively searches for the best parameter that optimizes the chosen
#     parameter
# 3) The main purpose of splitting our data into train and test is to see how well our model
#     generalizes to the unseen data. We are really no way interested in the accuracies on training
#     data, but on the new unseen data. Although I used train_test_split function before, there 
#     are more robust ways of assessing how well the model generalizes than just doing a single 
#     split of the data. It's a good practice to split the data into training, validation and test 
#     set. The model being trained on a training set is tuned to increase its performance on 
#     the validation set, and then the final efficiency is checked on the test set. 
#     Among the various cross-validation strategies, I decided to go with the StratifiedShuffleSplit, 
#     and the main reason of chosing it is because the our data is small, and we do want to have
#     the right representation of both the target and independent variables in the split. 
#     
#     We will check various parameters against each model:
#     1) Model Generalizability (Mean f1 score)
#     2) The Best Cross Validation Accuracy 
#     3) The Best Parameters
#     4) The Best Estimator Score
#     5) The Classification Report

features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments',
                 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 
                 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 
                 'total_stock_value', 'to_messages', 'from_messages', 'from_this_person_to_poi', 
                 'from_poi_to_this_person', 'shared_receipt_with_poi', 'fraction_from_poi', 'fraction_to_poi']


data_dict = enronData.to_dict(orient='index')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
y, X = targetFeatureSplit(data)
X = np.array(X)
y = np.array(y)

### Cross-validation
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

SCALER = [None, StandardScaler()]
SELECTOR__K = [10, 13, 15, 18, 'all']
REDUCER__N_COMPONENTS = [2, 4, 6, 8, 10]


'''
Gaussian Naive-Bayes
'''

pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', GaussianNB())
    ])

param_grid = {
    'scaler': SCALER,
    'selector__k': SELECTOR__K,
    'reducer__n_components': REDUCER__N_COMPONENTS
}

gnb_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
gnb_grid.fit(X, y)
evaluate_model(gnb_grid, X, y, sss)
print(gnb_grid.best_estimator_)
clf = gnb_grid.best_estimator_
test_classifier(gnb_grid.best_estimator_, my_dataset, features_list)


'''
Feature Importance
'''

kbest = gnb_grid.best_estimator_.named_steps['selector']

features_array = np.array(features_list)
features_array = np.delete(features_array, 0)
indices = np.argsort(kbest.scores_)[::-1]
k_features = kbest.get_support().sum()

features = []
for i in range(k_features):
    features.append(features_array[indices[i]])

features = features[::-1]
scores = kbest.scores_[indices[range(k_features)]][::-1]

plt.barh(range(k_features), scores)
plt.yticks(np.arange(0.4, k_features), features)
plt.title('Feature Importances Using KBest')
plt.show()

'''
Dumping Gaussian Naive Bayes
''''

CLF_PICKLE_FILENAME = "my_classifier_gaussian.pkl"
DATASET_PICKLE_FILENAME = "my_dataset_gaussian.pkl"
FEATURE_LIST_FILENAME = "my_feature_list_gaussian.pkl"

my_dump_classifier_and_data(clf,my_dataset,features_list)


'''
SVM 
'''

C_PARAM = np.logspace(-2, 3, 6)
GAMMA_PARAM = np.logspace(-4, 1, 6)
CLASS_WEIGHT = ['balanced', None]
KERNEL = ['rbf', 'sigmoid']

pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', SVC())
    ])

param_grid = {
    'scaler': SCALER,
    'selector__k': SELECTOR__K,
    'reducer__n_components': REDUCER__N_COMPONENTS,
    'classifier__C': C_PARAM,
    'classifier__gamma': GAMMA_PARAM,
    'classifier__class_weight': CLASS_WEIGHT,
    'classifier__kernel': KERNEL
}

svc_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)

evaluate_model(svc_grid, X, y, sss)

test_classifier(svc_grid.best_estimator_, my_dataset, features_list)

clf_svc = svc_grid.best_estimator_

'''
Dumping SVM
'''


CLF_PICKLE_FILENAME = "my_classifier_svc.pkl"
DATASET_PICKLE_FILENAME = "my_dataset_svc.pkl"
FEATURE_LIST_FILENAME = "my_feature_list_svc.pkl"

my_dump_classifier_and_data(clf_svc,my_dataset,features_list)


'''
Decision Tree
'''

CRITERION = ['gini', 'entropy']
SPLITTER = ['best', 'random']
MIN_SAMPLES_SPLIT = [2, 4, 6, 8]
CLASS_WEIGHT = ['balanced', None]

pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', DecisionTreeClassifier())
    ])

param_grid = {
    'scaler': SCALER,
    'selector__k': SELECTOR__K,
    'reducer__n_components': REDUCER__N_COMPONENTS,
    'classifier__criterion': CRITERION,
    'classifier__splitter': SPLITTER,
    'classifier__min_samples_split': MIN_SAMPLES_SPLIT,
    'classifier__class_weight': CLASS_WEIGHT,
}

tree_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
print('working')

evaluate_model(tree_grid, X, y, sss)
print('working')

test_classifier(tree_grid.best_estimator_, my_dataset, features_list)

clf_tree = tree_grid.best_estimator_

'''
Dumping Decision Tree
'''
CLF_PICKLE_FILENAME = "my_classifier_tree.pkl"
DATASET_PICKLE_FILENAME = "my_dataset_tree.pkl"
FEATURE_LIST_FILENAME = "my_feature_list_tree.pkl"

my_dump_classifier_and_data(clf_tree,my_dataset,features_list)

