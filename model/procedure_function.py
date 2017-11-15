import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, GroupKFold
from sklearn.externals import joblib

import pandas as pd
import os

main_dir = os.path.split(os.getcwd())[0]
result_dir = main_dir + '/results'

def shuffle_data(x,y,nb_persub):
    # shuffle each subject's data
    # x is n subjects' data
    x_shuffle = np.empty(x.shape)
    y_shuffle = np.empty(y.shape)
    ids_shuffle = []
    for i in range(int(x.shape[0] / nb_persub)):
        start = i * nb_persub
        end = (i + 1) * nb_persub
        index_shuffle = [j for j in range(start, end)]
        np.random.shuffle(index_shuffle)
        ids_shuffle.append(index_shuffle)
        x_shuffle[start:end] = x[index_shuffle]
        y_shuffle[start:end] = y[index_shuffle]
    ids_shuffle = np.vstack(ids_shuffle)
    # np.save(result_dir + 'datasave/shuffle_ids_{}_{}_{}_{}r_{}subs_{}trainsize_{}totalrounds.npy'.
    #         format(experiment, clf_method, data_method, r, k, trainsize, nrounds),
    #         ids_shuffle)

    return x_shuffle, y_shuffle

def split_subjects(subjects):
    # separate subjects
    index_subjects = []
    for subject in np.unique(subjects):
        index = [j for j, s in enumerate(subjects) if s == subject]
        index_subjects.append(index)
    return index_subjects

def gridsearch(x, y_target, subjects, cross_v, experiment, clf_method,
               nor_method, cv_start, njobs=1):

    note = '{}_{}_{}_{}cvstart'.format(experiment, nor_method, clf_method,
                                          cv_start)
    svm_parameters = [{'kernel': ['linear'],
                       'C': [10, 1, 0.1, 0.01, 0.005, 0.001, 0.0008,
                               0.0005, 0.0001, 0.00005, 0.00001]}]

    logis_parameters = [{'penalty': ['l1', 'l2'],
                         'C': [10, 1, 0.1, 0.01, 0.005, 0.001, 0.0008,
                               0.0005, 0.0001, 0.00005, 0.00001]}]
    clf = LogisticRegression if clf_method == 'logis' else SVC
    params = logis_parameters if clf_method == 'logis' else svm_parameters

    train, test = cross_v
    x_train = x[train]
    y_train = y_target[train]
    x_test = x[test]
    y_test = y_target[test]
    subjects_train = subjects[train]
    if experiment == 'meg':
        # gkf = GroupKFold(n_splits=7)
        # grid_cv = list(gkf.split(x_train, y_train, subjects_train))
        logo = LeaveOneGroupOut()
        grid_cv = list(logo.split(x_train, y_train, subjects_train))

    else:
        # leave 2 subjects out for fmri in gridsearch
        gkf = GroupKFold(n_splits=45)
        grid_cv = list(gkf.split(x_train, y_train, subjects_train))

        # logo = LeaveOneGroupOut()
        # grid_cv = list(logo.split(x_train, y_train, subjects_train))
        # print('logo')
    grid_clf = GridSearchCV(clf(), params, cv=grid_cv, n_jobs=njobs)
    grid_clf.fit(x_train, y_train)



    print('best params', grid_clf.best_params_)
    joblib.dump(grid_clf.best_params_,
                result_dir + 'gridtables/{}cv_gridbest.pkl'.format(note))
    # grid_bestpara = joblib.load(result_dir + 'joblib.pkl')
    grid_csv = pd.DataFrame.from_dict(grid_clf.cv_results_)
    with open(result_dir + '/gridtables/{}cv_gridtable.csv'.format(note), 'w') as f:
        grid_csv.to_csv(f)

    pre = grid_clf.predict(x_test)
    score_op = accuracy_score(y_test, pre)
    print('optimal transport accuracy of {}th split:'.format(cv_start), score_op)

    return score_op
