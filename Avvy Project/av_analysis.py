# Evan Anderson
# CDA Project - Avalanche Prediction
# CDA Fall 2021
# December 11th, 2021

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as accuracy
from sklearn.svm import OneClassSVM as ocsvm
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, Lasso, lasso_path
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LinearRegression

import copy
from sklearn.preprocessing import MinMaxScaler
from itertools import cycle
import matplotlib.pyplot as plt

path = 'Data/av_data.csv'
avy_data = pd.read_csv(path)
# drop rows with NaNs
avy_data = avy_data.dropna()


# Columns
all_cols = ['Date', 'Air_Temp_Avg_degF', 'Air_Temp_Max_degF', 'Air_Temp_Min_degF',
       'Precip_Accum_in_', 'Precip_Increment_in', 'Precip_Increment_Snow_in',
       'Precip_MTD_in', 'Snow_Depth_in', 'Snow_Density_pct', 'Snow_Rain_Ratio',
       'percip_prior_2days', 'percip_prior_4days', 'percip_prior_7days',
       'percip_prior_14days', 'temp_min_prior_2days', 'temp_min_prior_4days',
       'temp_min_prior_7days', 'temp_min_prior_14days', 'temp_max_prior_2days',
       'temp_max_prior_4days', 'temp_max_prior_7days', 'temp_max_prior_14days',
       'snow_density_avg_prior_2days', 'snow_density_avg_prior_4days',
       'snow_density_avg_prior_7days', 'snow_density_avg_prior_14days',
       'av_bol']

pred_cols = ['Air_Temp_Avg_degF', 'Air_Temp_Max_degF', 'Air_Temp_Min_degF',
       'Precip_Accum_in_', 'Precip_Increment_in', 'Precip_Increment_Snow_in',
       'Precip_MTD_in', 'Snow_Depth_in', 'Snow_Density_pct', 'Snow_Rain_Ratio',
       'percip_prior_2days', 'percip_prior_4days', 'percip_prior_7days',
       'percip_prior_14days', 'temp_min_prior_2days', 'temp_min_prior_4days',
       'temp_min_prior_7days', 'temp_min_prior_14days', 'temp_max_prior_2days',
       'temp_max_prior_4days', 'temp_max_prior_7days', 'temp_max_prior_14days',
       'snow_density_avg_prior_2days', 'snow_density_avg_prior_4days',
       'snow_density_avg_prior_7days', 'snow_density_avg_prior_14days']

X = avy_data[pred_cols]
y = avy_data['av_bol']
# print(X.head())

# Shuffling data
X, y = shuffle(X, y, random_state=24)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=24)

RandomForestAvy = False
if RandomForestAvy:
    clf_rf = RandomForestClassifier(
        # warm_start=True,
        # oob_score=True,
        random_state=24)

    max_estimators = 150
    rf_error = []
    if False:
        for i in range(1, max_estimators + 1):
            clf_rf.set_params(n_estimators=i)
            clf_rf.fit(X_train, y_train)
            y_pred_rf = clf_rf.predict(X_test)
            rf_a = 1 - accuracy(y_test, y_pred_rf)
            rf_error.append([i, rf_a])

        rf_error = pd.DataFrame(rf_error)
        plt.plot(rf_error.iloc[:, 0], rf_error.iloc[:, 1], label="RF Error")
        plt.legend()
        plt.show()
    clf_rf.set_params(n_estimators=37)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    rf_a = 1 - accuracy(y_test, y_pred_rf)
    print(rf_a)
    print(clf_rf.score(X_test, y_test))
    print(metrics.confusion_matrix(y_test, y_pred_rf))


SVMAvy = False
if SVMAvy:
    if False:
        for n in range (7,24):
            pca = PCA(n_components=n)
            print("PCA comps: {}".format(n))
            X_new = pca.fit_transform(X, y)

            X, y = shuffle(X, y, random_state=24)
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20, random_state=24)

            min_error = 100
            min_i = 0


            for i in np.linspace(1, 500, 50):
                clf_svm = ocsvm(kernel='rbf', gamma=1 / i)
                clf_svm.fit(X_train, y_train)
                y_pred_svm = clf_svm.predict(X_test)
                svm_error = mse(y_test, y_pred_svm)

                # print('gamma = 1/{}: '.format(i), svm_error)
                if min_error > svm_error:
                    min_error = svm_error
                    min_i = 1 / i

            print('Best Gamma: ', min_i)
            print('Mini Error: ', min_error)

    if True:
        pca = PCA(n_components=11)
        print("PCA comps: {}".format(11))
        pca.fit_transform(X, y)

        print(pca.components_)

        X_new = pca.fit_transform(X, y)

        X, y = shuffle(X, y, random_state=24)
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20, random_state=3)

        clf_svm = ocsvm(kernel='rbf', gamma=1/47)
        clf_svm.fit(X_train, y_train)
        y_pred_svm = clf_svm.predict(X_test)
        y_pred_svm = np.where(y_pred_svm == -1, 0, y_pred_svm)
        svm_error = mse(y_test, y_pred_svm)
        svm_accuracy = 1 - accuracy(y_test, y_pred_svm)
        print(svm_accuracy)
        print(metrics.confusion_matrix(y_test, y_pred_svm))


LogRegAvy = False
if LogRegAvy:
    pca = PCA(n_components=6)
    X_new = pca.fit_transform(X, y)

    X, y = shuffle(X, y, random_state=24)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20, random_state=24)

    clf_lrcv = LogisticRegressionCV(cv=5, random_state=24)
    clf_lrcv.fit(X_train, y_train)
    lr_preds = clf_lrcv.predict(X_test)
    print(max(lr_preds))
    print(min(lr_preds))
    # print(y_test)

    fig, ax = plt.subplots()
    ax.plot(y_test, lr_preds)
    plt.show()

        # plt.figure

        # for threshold in np.linspace(0,1,6):
        #     lr_binary_preds = lr_preds > threshold
        #     lr_acc = metrics.accuracy_score(y_test, lr_binary_preds)
        #     print("LinearRegression Acc for Threshold {}: ".format(threshold), lr_acc)
        #     lr_prec = metrics.precision_score(y_test, lr_binary_preds)
        #     print("LinearRegression Precision for Threshold {}: ".format(threshold), lr_prec)
        #     lr_recall = metrics.recall_score(y_test, lr_binary_preds)
        #     print("LinearRegression Recall for Threshold {}: ".format(threshold), lr_recall)


LassoAvy =False
if LassoAvy:
    # Scaling the data:
    scale_cols = ['Air_Temp_Avg_degF', 'Air_Temp_Max_degF', 'Air_Temp_Min_degF',
       'Precip_Accum_in_', 'Precip_Increment_in', 'Precip_Increment_Snow_in',
       'Precip_MTD_in', 'Snow_Depth_in', 'Snow_Density_pct', 'Snow_Rain_Ratio',
       'percip_prior_2days', 'percip_prior_4days', 'percip_prior_7days',
       'percip_prior_14days', 'temp_min_prior_2days', 'temp_min_prior_4days',
       'temp_min_prior_7days', 'temp_min_prior_14days', 'temp_max_prior_2days',
       'temp_max_prior_4days', 'temp_max_prior_7days', 'temp_max_prior_14days',
       'snow_density_avg_prior_2days', 'snow_density_avg_prior_4days',
       'snow_density_avg_prior_7days', 'snow_density_avg_prior_14days']

    alphas = np.linspace(1, 10000, 10000)
    coefs = []

    clf = LassoCV(alphas=alphas, cv=5)
    clf.fit(X_train, y_train)
    print(clf.alpha_)
    print(X_train.columns, clf.coef_)

    alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps=.00005)

    colors = cycle(["b", "r", "g", "c", "k"])
    log_alphas_lasso = np.log10(alphas_lasso)
    # neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_l, c in zip(coefs_lasso, colors):
        l1 = plt.plot(log_alphas_lasso, coef_l, c=c)
        # l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle="--", c=c)

    plt.xlabel("Log(alpha)")
    plt.ylabel("coefficients")
    plt.title("Lasso Paths")
    # plt.legend(("Lasso", "Elastic-Net"), loc="lower left")
    plt.axis("tight")
    plt.show()

    print(clf.coef_)


BayesianAvy = False
if BayesianAvy:
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_preds = nb.predict(X_test)
    print(nb.score(X_test, y_test))
    print(metrics.confusion_matrix(y_test, nb_preds))
    print('Total Avs: ', y_test.sum())


CARTAvy = True
if CARTAvy:
    clf = DecisionTreeClassifier(random_state=2
                                 # , max_depth=5
                                ).fit(X_train, y_train)
    fig = plt.figure(figsize=(50,20))
    plot_tree(clf, filled=True)
    fig.savefig("images/cart_decistion_tree.png")

    cart_preds = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    print(metrics.confusion_matrix(y_test, cart_preds))
    print('Total Avs: ', y_test.sum())

KNNAvy = False
if KNNAvy:
    for k in range(2,7):
        clf = KNN(n_neighbors=k, weights='distance').fit(X_train, y_train)
        knn_preds = clf.predict(X_test)
        print('\n K = {}'.format(k))
        print(clf.score(X_test, y_test))
        print(metrics.confusion_matrix(y_test, knn_preds))
        print('Total Avs: ', y_test.sum())

LinReg = False
if LinReg:
    pca = PCA(n_components=20)
    X_new = pca.fit_transform(X, y)

    X, y = shuffle(X, y, random_state=24)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20, random_state=24)

    clf = LinearRegression(normalize=True).fit(X_train, y_train)
    linreg_preds = clf.predict(X_test)

    # splitting outputs
    zero_noav = []
    one_av = []
    for pred, label in zip(linreg_preds, y_test):
        if label == 0:
            zero_noav.append(pred)
        else:
            one_av.append(pred)


    fig, ax = plt.subplots()
    n_bins = 20
    ax.hist([zero_noav, one_av], n_bins, density=True, histtype='bar', stacked=True)
    plt.show()

#TODO
AspectPrediction = False
