from __future__ import division

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pickle
import cv2

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score


def cm_report(actual, predicted, print_report=True):
    '''Confusion Matrix Report takes a test set, and testset predictions;
    prints a report on value counts and classification percentages
    returns the confusion matrix and 'percentage' matrix'''

    # groups = sorted(labels.value_counts().index)
    groups = sorted(np.unique(actual))
    cm = confusion_matrix(actual, predicted, labels=groups)
    alphabetized_counts = np.array(
        [[actual.value_counts().get(group, default=0) for group in groups]])

    percent_matrix = cm*100/alphabetized_counts.T
    percent_matrix = percent_matrix.round(decimals=2)
    percent_matrix = np.nan_to_num(percent_matrix)

    if print_report:
        # print "Training value counts: \n", labels.value_counts()
        print "\nTest set value counts:\n", actual.value_counts()
        print "\n"

        # TODO: how to make rows print with common spacing
        # for g,c in zip(groups,cm):
        #     print g, "\t\t", c
        # for g,p in zip(groups,percent_matrix):
        #     print g, "\t\t", p

        for g in groups:
            print g
        print '\nTest Confusion Matrix:\n', cm
        print '\nPercentage True in each predicted class\n', percent_matrix
        print '\n'

    return cm, percent_matrix


def eval_model(model, X_test, y_test):
    """
    prints a variety of performance metrics for a griven model
    """
    def crossval(score_type):
        try:
            score = cross_val_score(model, X_test, y_test,
                                    cv=5, scoring=score_type)
        except:
            score = 'invalid metric'

        return score

    print "accuracy : ", crossval('accuracy')
    print "precision: ", crossval('precision')
    print "recall   : ", crossval('recall')
    print "- logloss: ", crossval('neg_log_loss')


def plot_probas(predicted, save_as=None):
    """
    plot_probas takes predictions for a single image
    and plots a horizontal bar chart showing
    the predicted probability that the image belongs to each class
    """
    #predicted = [('Sit',0.36),('Stand',0.54),('Lie',0.08)]

    #unzip into value and labels
    labels, val = zip(*predicted)[0],zip(*predicted)[1]

    # highest precicted category and its probability
    pred, proba = labels[-1], val[-1]

    fig, ax = plt.subplots()
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, val, align='center', color='#6982A0', alpha = 0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, rotation=50)
    ax.set_xlabel('Probability')
    ax.set_xlim([0,100])
    ax.set_title('Prediction: {}  --  Probability: {}'.format(pred, proba))
    ax.grid(True)

    if save_as:
        save_loc = '{}.png'.format(save_as)
        print "\nSaving to {}\n".format(save_loc)
        plt.savefig(save_loc)#, bbox_inches='tight')

    plt.show()
    # plt.clf() # use in web app to prevent overlaying new plots


def get_image(path):
    """
    Fetches an image with opencv,
    prepares image to plot in matplotlib
    """
    img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.cv.CV_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rotate_bound(image, angle):
    """
    rotates an image
    Thankyou jrosebr1 on github for imutils
    https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def show_img(path, title = None, rotation=None):
    """
    uses matplotlib to plot an image
    """
    img = get_image(path)

    if rotation:
        img = rotate_bound(img, rotation)

    # scale_factor = 1.2
    # plt.figure(figsize=(6*scale_factor,4*scale_factor))
    plt.subplot(111)
    plt.imshow(img, cmap='gray')
    plt.suptitle(title)
    plt.show()


def plot_confusion_matrix(cm, classes,
                        #   normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_as=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`,
    Plot can be saved to 'data/plots/<name>.png' by setting save_as='<name>'
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    #
    # print(cm,'\n')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_as:
        plt.savefig('data/plots/{}.png'.format(save_as),
                    bbox_inches='tight')
    plt.show()
