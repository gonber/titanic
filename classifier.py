# preprocessing is based on myfirstforest.py as provided by Kaggle

import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt

def Draw(feature1, feature2, labels):
    colors = ["r", "b"]
    for ii, pp in enumerate(features):
        plt.scatter(feature1[ii], feature2[ii], color=colors[int(labels[ii])])

    plt.xlabel("feature1")
    plt.ylabel("feature2")
    plt.show()

def loadData(csv, drop_features):
    df = pd.read_csv(csv, header=0)

    # female = 0, Male = 1
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    df = df.drop(['Sex'], axis=1)

    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.
    # All missing Embarked -> just make them embark from most common place
    if len(df.Embarked[df.Embarked.isnull()]) > 0:
        df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values

    Ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
    df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    # All the ages with no data -> make the median of all Ages
    median_age = df['Age'].dropna().median()
    if len(df.Age[df.Age.isnull()]) > 0:
        df.loc[(df.Age.isnull()), 'Age'] = median_age

    # All the missing Fares -> assume median of their respective class
    if len(df.Fare[df.Fare.isnull()]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = df[df.Pclass == f+1]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            df.loc[(df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]

    def binning(x):
        if x < 10:
            return 3
        if x < 20:
            return 2
        return 1
    df.Fare = df.Fare.map( lambda x: binning(x)).astype(int)

    # Collect the test data's PassengerIds before dropping it
    ids = df['PassengerId'].values

    df = df.drop(drop_features, axis=1)

    return (df.values, df.columns.values, ids)

drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'SibSp', 'Parch']
train_data, train_name_features, train_ids = loadData('train.csv', drop)
test_data, test_name_features, test_ids = loadData('test.csv', drop)

features = train_data[0::,1::]
labels = train_data[0::,0]
# Draw(train_ids, features[0::,3], labels)

print 'Training...'

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

# pca = RandomizedPCA(n_components=2)
# pca.fit(features_train)
# print pca.explained_variance_ratio_
# features_train = pca.transform(features_train)
# features_test = pca.transform(features_test)
# test_data = pca.transform(test_data)

params1 = {'min_samples_split': 25,
           'n_estimators': 15,
           'random_state': 42}
clf1 = RandomForestClassifier(**params1)

params2 = {'base_estimator': DecisionTreeClassifier(min_samples_split=25),
           'learning_rate': 0.01,
           'n_estimators': 15,
           'random_state': 42}
clf2 = AdaBoostClassifier(**params2)

params3 = {'learning_rate': 0.5,
           'max_depth': 5,
           'min_samples_split': 40,
           'n_estimators': 3,
           'random_state': 42}
clf3 = GradientBoostingClassifier(**params3)

params4 = {'n_neighbors': 22}
clf4 = KNeighborsClassifier(**params4)

params5 = {'min_samples_split': 22}
clf5 = DecisionTreeClassifier(**params5)

params6 = {'C': 0.3}
clf6 = LogisticRegression(**params6)

params7 = {'C': 0.6,
           'gamma': 1.2}
clf7 = SVC(**params7)

params8 = {'C': 0.04,
           'max_iter': 4,
           'random_state': 42}
clf8 = LinearSVC(**params8)

clfe = VotingClassifier(estimators=[('rf', clf1), ('ab', clf2),
    ('gbo', clf3), ('kn', clf4), ('dt', clf5), ('lr', clf6),
    ('sv', clf7), ('lsv', clf8)],
    voting='hard')

cv = StratifiedShuffleSplit(labels, n_iter=100, test_size=0.2, random_state=42)

# clf = GridSearchCV(estimator=clf8, param_grid=params8, cv=cv)
# clf.fit(features, labels)
# print clf.best_score_
# print clf.best_params_
# abc

clf = clfe

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx, test_idx in cv:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

    ### fit the classifier using training set, and test on test set
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print "Warning: Found a predicted label not == 0 or 1."

total_predictions = true_negatives + false_negatives + false_positives + true_positives
accuracy = 1.0*(true_positives + true_negatives)/total_predictions
precision = 1.0*true_positives/(true_positives+false_positives)
recall = 1.0*true_positives/(true_positives+false_negatives)
f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)

print 'Predicting...'
clf.fit(features, labels)
output = clf.predict(test_data).astype(int)

predictions_file = open("classifier.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(test_ids, output))
predictions_file.close()
print 'Done.'
