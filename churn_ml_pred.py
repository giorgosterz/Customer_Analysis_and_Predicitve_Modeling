# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:41:00 2020

@author: giorg
"""



########################################################## DATA PREPROCESSING ######################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')

old_age = dataset[dataset['Age']>=75].index
dataset.drop(old_age,inplace=True)

products = dataset[dataset['NumOfProducts']>=3].index
dataset.drop(products,inplace=True)


sns.countplot(x='Exited', data = dataset)

# Seperate independent variales with target variable
X = dataset.iloc[: ,:-1].values
y = dataset.iloc[: , -1].values

# Keeping the columns we are interested
X = np.delete(X, [0,1,2],axis=1) # Droping RowNumber, CustomerID, Surname

# Encoding the categorical variables Country and Gender

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1,2])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2, train_size = 0.8, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

###################################################### LINEAR OR NON-LINEAR DATA? #################################################
# Building the SVM classifier 
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', gamma = 'auto', random_state = 0)
classifier.fit(X_train,y_train) 

# Grid search to figure out if our problem is linear or non-linear
from sklearn.model_selection import GridSearchCV
parameters = [{'kernel': ['linear']},
               {'kernel': ['rbf']}
               ]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_   #Best kernel is rbf so our problem is non linear

print(best_accuracy, best_parameters)


########################################### ROC CURVE ANALYSIS FOR OUR NON-LINEAR MODELS ###########################################
# Import Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Splitting our training set
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=6)


# Defining a list containing the classifiers    
classifiers = [ 
               GaussianNB(), 
               KNeighborsClassifier(), 
               DecisionTreeClassifier(criterion = 'entropy', random_state = 1234),
               RandomForestClassifier(criterion = 'entropy',random_state=1234),
               SVC(kernel = 'rbf', gamma = 'auto',probability=True, random_state = 1234),
               XGBClassifier(random_state = 1234)]


# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', "mean fpr","mean tpr",'mean auc'])

for classifier in classifiers:
    
    # Creating emptylists to hold the results
    fpr_s = []
    tpr_s = []
    auc_s = []
    
    # Iterate through the splits and collect False Positives and True Negatives
    for (train, test) in (cv.split(X_train, y_train)):
        
        model = classifier.fit(X_train[train],y_train[train])
        yproba = model.predict_proba(X_train[test])[:,1]
        fpr, tpr, _ = roc_curve( y_true=y_train[test], y_score=yproba )
        fpr_s.append(fpr)
        tpr_s.append(tpr)
        
        auc = roc_auc_score( y_true=y_train[test], y_score=yproba )
        auc_s.append(auc)
        
    
    # Taking the sum for each element of each list    
    fpr_s_sum = [sum(i) for i in zip(*fpr_s)]
    tpr_s_sum = [sum(i) for i in zip(*tpr_s)]
    auc_s_sum = sum(auc_s)
    
    # Taking the mean value for each element of the list
    mean_fpr_s = [x / len(fpr_s) for x in fpr_s_sum]
    mean_tpr_s = [x / len(tpr_s) for x in tpr_s_sum]
    mean_auc_s = sum(auc_s) / len(auc_s)
     
    # Appending results to the dataframe
    result_table = result_table.append({'classifiers':classifier.__class__.__name__,
                                        "mean fpr":mean_fpr_s, 
                                        "mean tpr":mean_tpr_s, 
                                        'mean auc':mean_auc_s}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

# Ploting the ROC Curves

fig = plt.figure(figsize=(12,10))

for i in result_table.index:
    plt.plot(result_table.loc[i]['mean fpr'], 
             result_table.loc[i]['mean tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['mean auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=20)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=20)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=25)
plt.legend(prop={'size':13}, loc='lower right')
plt.legend(fontsize=20) # using a size in points
plt.legend(fontsize="x-large") # using a named size

plt.show()


#------------ Creating the metrics table through cross validation ------------
from sklearn.model_selection import cross_validate
metrics_table = pd.DataFrame(columns=['classifiers', "mean accuracy",'mean precision',"mean recall",'mean f1'])

for classifier in classifiers:
    
    scoring = ['accuracy', 'precision','recall','f1']
    scores = cross_validate(classifier, X_train, y_train, scoring=scoring,cv=5)
    #print((scores.keys()))

    metrics_table = metrics_table.append({'classifiers':classifier.__class__.__name__,
                                          'mean accuracy': scores['test_accuracy'].mean(),
                                          'mean precision': scores['test_precision'].mean(),
                                          'mean recall': scores['test_recall'].mean(),
                                          'mean f1': scores['test_f1'].mean()}, ignore_index=True)
    
   
metrics_table.set_index('classifiers', inplace=True)

#----- Tuning the best perfofming model ----
xgb_cls = XGBClassifier(random_state = 1234)
from sklearn.model_selection import GridSearchCV

xgb_parameters = [{'eta':[0.1,0.2,0.3,0.4,0.5,1],
                   'gamma':[0,0.2,0.4,0.6,0.8,1],
                   'max_depth':[1,2,3,4,5,6]}]
grid_search = GridSearchCV(estimator = xgb_cls, param_grid = xgb_parameters, scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
# Taking the results
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#----------------------- Making the Prediction ----------------------------

final_cls = XGBClassifier(eta = 0.5 , gamma = 0.6 , max_depth = 5)
final_cls.fit(X_train,y_train)
y_pred = final_cls.predict(X_test)

# Making the Confusion Matrix of the final model
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import matplotlib.pyplot as plt
plot_confusion_matrix(final_cls, X_test, y_test,
                                 labels=dataset["Exited"].unique(),
                                 cmap=plt.cm.Blues,
                                 values_format = '.1g',
                                 normalize = 'all'
                                 )
plt.grid(False)

# Creating the ROC curve
from sklearn.metrics import roc_curve, plot_roc_curve
plot_roc_curve(final_cls,X_test,y_test)


# Creating a table with metrics for winning alg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

winners_metrics_table = pd.DataFrame(columns=['classifier', "accuracy",'precision',"recall",'f1'])
winners_metrics_table = winners_metrics_table.append({'classifier':final_cls.__class__.__name__,
                                          'accuracy': accuracy_score(y_test,y_pred),
                                          'precision': precision_score(y_test,y_pred),
                                          'recall': recall_score(y_test,y_pred),
                                          'f1': f1_score(y_test,y_pred)}, ignore_index=True)
winners_metrics_table.set_index('classifier', inplace=True)


# ---------------------- Creating an ANN to compare with ---------------------------
from keras.models import Sequential
from keras.layers import Dense

ANN_model = Sequential()

# The Input Layer :
ANN_model.add(Dense(128, kernel_initializer='uniform',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
ANN_model.add(Dense(256, kernel_initializer='uniform',activation='relu'))
ANN_model.add(Dense(256, kernel_initializer='uniform',activation='relu'))
ANN_model.add(Dense(256, kernel_initializer='uniform',activation='relu'))

# The Output Layer :
ANN_model.add(Dense(1, kernel_initializer='uniform',activation='sigmoid'))

#Compiling the ANN (ousiastika applying the Stochastic Gradient Descent)
ANN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
ANN_model.summary()

# Fitting the ANN to the Training set
ANN_model.fit(X_train, y_train, batch_size=10 ,epochs=10)

# Predicting the Test set results
y_pred_ann = ANN_model.predict(X_test)
y_pred_ann = (y_pred_ann>0.5)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_ann = confusion_matrix(y_test, y_pred_ann)

# Metrics tabe for ANN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ANN_metrics_table = pd.DataFrame(columns=['classifier', "accuracy",'precision',"recall",'f1'])
ANN_metrics_table = ANN_metrics_table.append({'classifier':'ANN',
                                          'accuracy': accuracy_score(y_test,y_pred_ann),
                                          'precision': precision_score(y_test,y_pred_ann),
                                          'recall': recall_score(y_test,y_pred_ann),
                                          'f1': f1_score(y_test,y_pred_ann)}, ignore_index=True)
ANN_metrics_table.set_index('classifier', inplace=True)