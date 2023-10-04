import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("C:/Users/jamie/Downloads/MachineLearningWithPython-master/MachineLearningWithPython-master/Notebooks/data/pima-data.csv")

#print(df.isnull().values.any())

# create function to display correlation in dataset
def plot_corr(df, size=11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    return fig, ax

fig, ax = plot_corr(df)

#plt.show()

# remove skin column due to extraneous correlation

del df['skin']

# check data types - must all be integers (true -> 1 false -> 0)

diabetes_map = {True : 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

# check true or false ratio

num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
print("Number of True cases: {0} ({1:2.2f}%)".format(num_true, (num_true/ (num_true + num_false)) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/ (num_true + num_false)) * 100))

# split the data into training and testing
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

x = df[feature_col_names].values
y = df[predicted_class_names].values
split_test_size = 0.3

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)

print("{0:0.2f}% in training set".format((len(x_train)/len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(x_test)/len(df.index)) * 100))

# inputing missing values with mean

fill_0 = SimpleImputer(missing_values=0, strategy="mean")

x_train = fill_0.fit_transform(x_train)
x_test_ = fill_0.fit_transform(x_test)

# training initial algorithm - Naive Bayes

nb_model = GaussianNB() # create Gaussian Naive Bayes model object
nb_model.fit(x_train, y_train.ravel())

# performance on training data 

nb_predict_train = nb_model.predict(x_train)

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))

# performance on testing data
nb_predict_test = nb_model.predict(x_test)

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))

# logistic regression to improve performance

lr_model = LogisticRegression(C=0.7, random_state=42)
lr_model.fit(x_train, y_train.ravel())
lr_predict_test = lr_model.predict(x_test)

# training metrics

# print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
# print(metrics.confusion_matrix(y_test, lr_predict_test) )
# print("")
# print("Classification Report")
# print(metrics.classification_report(y_test, lr_predict_test))

# Setting regularization parameter

# C_start = 0.1
# C_end = 5
# C_inc = 0.1

# C_values, recall_scores = [], []

# C_val = C_start
# best_recall_score = 0
# while C_val < C_end:
#     C_values.append(C_val)
#     lr_model_loop = LogisticRegression(C=C_val, random_state=42)
#     lr_model_loop.fit(x_train, y_train.ravel())
#     lr_predict_loop_test = lr_model_loop.predict(x_test)
#     recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
#     recall_scores.append(recall_score)
#     if recall_score > best_recall_score:
#         best_recall_score = recall_score
#         best_lr_predict_test = lr_predict_loop_test

#     C_val += C_inc

# best_score_C_val = C_values[recall_scores.index(best_recall_score)]
# print("1st max value of {0:3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

#%matplotlib inline
# plt.figure()
# plt.plot(C_values, recall_scores, "-")
# plt.xlabel("C value")
# plt.ylabel("recall_score")
# plt.show()

# Setting regularization parameter

C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while C_val < C_end:
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, class_weight="balanced", random_state=42)
    lr_model_loop.fit(x_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(x_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if recall_score > best_recall_score:
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test

    C_val += C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
# print("1st max value of {0:3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

lr_model = LogisticRegression(class_weight="balanced", C=best_score_C_val, random_state=42)

# print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, best_lr_predict_test)))
# print(metrics.confusion_matrix(y_test, best_lr_predict_test))
# print("")
# print("Classification Report")
# print(metrics.classification_report(y_test, best_lr_predict_test))

