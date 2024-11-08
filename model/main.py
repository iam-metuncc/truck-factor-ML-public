import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


### just change variables below and run
def main(path):

    clfs = []
    clfs.append(RandomForestClassifier(random_state=3,criterion='entropy', n_estimators = 50, max_depth = 10, min_samples_split = 5,min_samples_leaf = 1, max_features = 'sqrt'))

    algorithms = ['knn', 'nb', 'dtree', 'rforest', 'logisticregression']


    df = pd.read_csv(path)
    print("Loaded Dataset\n", df.head())

    print("Dataset information")
    print(df.info())

    ## checking for null values
    print("Any null values in the dataset:", df.isnull().values.any())

    ## splitting the data into test and train.
    repoList = df['repo'].unique()

    X = df.drop(['author', 'repo', 'developer_random_id'], axis=1)
    y = df[['author']]
    print(X.head())
    print(y.head())
    dev_count = []
    for repo in repoList:
        dev_count.append(len(df[df['repo'] == repo].index))
    dev_count_boundary = np.quantile(sorted(dev_count), [0.90])
    index = 0
    for clf in clfs:
        print(algorithms[index])
        sensitivity_total = []
        precision_total = []
        f1_total = []

        repo_dictionary = {}
        f_big = []
        ratio = 0
        for repo in repoList:
            dfTrain = df[df['repo'] != repo]
            dfTest = df[df['repo'] == repo]
            X = dfTrain.drop(['author', 'repo', 'developer_random_id'], axis=1)
            y = dfTrain[['author']]
            X2 = dfTest.drop(['author', 'repo', 'developer_random_id'], axis=1)
            y2 = dfTest[['author']]

            train_X = X.to_numpy()
            train_y = y.to_numpy().reshape((-1,))

            test_X = X2.to_numpy()
            test_y = y2.to_numpy().reshape((-1,))

            clf.fit(train_X, train_y)
            pred_y = clf.predict(test_X)
            if (len(test_y) > dev_count_boundary):
                 x  = GaussianNB(priors=[0.05,0.95])
                 x.fit(train_X, train_y)
                 pred_y = x.predict(test_X)

            cf_matrix = confusion_matrix(test_y, pred_y, labels=[0, 1])
            TP = cf_matrix[1][1]
            TN = cf_matrix[0][0]
            FP = cf_matrix[0][1]
            FN = cf_matrix[1][0]

            if (TP == 0 and FN == 0):
                sensitivity = 0
            else:
                sensitivity = TP / (TP + FN)

            sensitivity_total += [sensitivity]

            if (TP == 0 and FP == 0):
                precision = 0
            else:
                precision = TP / (TP + FP)
            precision_total += [precision]
            if (precision == 0 or sensitivity == 0):
                f1 = 0
            else:
                f1 = (precision * sensitivity * 2) / (sensitivity + precision)

            print(cf_matrix)
            repo_dictionary[repo] = f1
            f1_total += [f1]
            if (len(test_y) > dev_count_boundary):
                ratio += (TP + FN) / (float(TP + FP + FN + TN))
                f_big.append(f1)


        repo_dictionary = sorted(repo_dictionary.items(), key=lambda x: x[1])
        f = np.array(f1_total).mean()


        print(f'algorithm: {algorithms[index]}\n'
              f'Average precision: {np.array(precision_total).mean()}\n'
              f'Average recall: {np.array(sensitivity_total).mean()}\n'
              f'Average f-score: {np.array(f1_total).mean()}')
        index += 1

        x = np.quantile(f1_total, [0.25, 0.5, 0.75])
        y = np.quantile(precision_total, [0.25, 0.5, 0.75])
        z = np.quantile(sensitivity_total, [0.25, 0.5, 0.75])

        print(f'fscore: {x}')
        print(f'precision: {y}')
        print(f'recall: {z}')


if __name__ == '__main__':
    main("github_ratio_anonymous.csv")
