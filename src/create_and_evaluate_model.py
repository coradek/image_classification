import os
import sys # needed?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import cPickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import cv2

from src import build_data as data
# from src import LogisticRegression_Multiclass as lrm
from src import evaluation as ev

class ModelMaker(object):
    """
    takes path to dir of photos (where sub folders are classes)
    gets tensorflow features
    creates and saves df
    trains logistic model
    evaluates model
    """
    def __init__(self, root_dir, image_directory):
        # Nested dir: each sub folder is a class
        self.root = root_dir
        self.dir = image_directory
        self.path = os.path.join(self.root, self.dir)
        self.df = None
        self.class_labels = None
        self.y = None
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_pred = None
        self.y_probas = None
        self.conf_matrix = None
        self.perc_matrix = None
        self.results = None
        self.missed_preds = None

    def process_images_and_train_model(self, split_state=None):
        # Todo: make this create df first time run
        #       and load df if already exists
        self.df = data.create_df(self.path)
        # self.df.to_csv('data/training_data.csv')

        df = self.df
        self.class_labels = np.unique(df.label)
        self.y = df['label']
        self.X = df.drop(df.columns[[2048, 2049]], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test =\
             train_test_split(self.X, self.y, test_size=0.3
                             , random_state=split_state)

        # These are all default values
        # (included incase I want to play with any of them)
        lr_model = LogisticRegression(penalty='l2', dual=False
                              , tol=0.0001, C=1.0, fit_intercept=True
                              , intercept_scaling=1, class_weight=None
                              , random_state=None, solver='liblinear'
                              , max_iter=100, multi_class='ovr'
                              , verbose=0, warm_start=False, n_jobs=-1)
        self.model = lr_model.fit(self.X_train, self.y_train)

    def get_test_result_dataframe(self):
        results = pd.DataFrame()
        results['true_label'] = self.y_test
        results['prediction'] = self.y_pred
        col_names = ['proba_'+c for c in self.class_labels]
        for c, p in zip(col_names, self.y_probas.T):
            results[c] = p
        results['file_path'] = self.df.loc[self.y_test.index.tolist()].file_path
        return results

    def evaluate_model(self):
        # may not be needed now that self.class_labels is added
        # labels = sorted(np.unique(self.df.labels))
        self.y_pred = self.model.predict(self.X_test)
        self.y_probas = self.model.predict_proba(self.X_test)

        self.results = self.get_test_result_dataframe()
        tru, pred = self.results.true_label, self.results.prediction
        self.missed_preds = self.results[tru != pred]

        c, p = ev.cm_report(self.y_test, self.y_pred, print_report=False)
        self.conf_matrix, self.perc_matrix = c, p

    def display_report(self):
        print("Training set cross validation metrics: ")
        print("\nLog loss:")
        print(cross_val_score(self.model, self.X_train, self.y_train
                              , groups=None, scoring='neg_log_loss'
                              , cv=None, n_jobs=-1, verbose=0, fit_params=None
                              , pre_dispatch='2*n_jobs'))

        print("\nAccuracy:")
        print(cross_val_score(self.model, self.X_train, self.y_train
                              , groups=None, scoring=None
                              , cv=None, n_jobs=-1, verbose=0, fit_params=None
                              , pre_dispatch='2*n_jobs'))

        print("\n#########")
        print("\nTest set metrics:")
        _, _ = ev.cm_report(self.y_test, self.y_pred)
        ev.plot_confusion_matrix(self.conf_matrix, classes=self.class_labels,
                                title='Confusion matrix')
        ev.plot_confusion_matrix(self.perc_matrix, classes=self.class_labels,
                                title='Percentage matrix')
