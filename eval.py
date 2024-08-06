import cv2
import numpy as np
from matplotlib import pyplot as plt

# scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from glob import glob
import os
import natsort



import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, roc_auc_score
import os
import yaml
import pdb
# from lib.config import parse_args
import warnings
import  natsort

warnings.filterwarnings("ignore")

"""
calculate metrics for entire retinal vessel images.
"""


def metrics(label_array, prediction_array, threshold_confusion = 0.5):
    """
    :param foreground: pixel value 255 is foreground.
    """
    # label_file_name = natsort.natsorted(os.listdir(label_path))
    # pred_file_name = natsort.natsorted(os.listdir(prediction_path))
    f1m = []
    accm = []
    aucm = []
    specificitym = []
    precisionm = []
    sensitivitym = []

    # pdb.set_trace()
    for i in range(len(label_array)):
        # label = Image.open(label_path + "/" + label_file_name[i])
        # label = label.resize((448,448))
        label = label_array[i][0]
        # label = cv2.resize(label,dsize = (224,224)).astype('uint8')


        # label[label <= 128] = 0
        # label[label > 128] = 1

        pred = prediction_array[i][0]  ##important
        # pred = pred.astype(np.uint8)
        pred = pred.flatten()/255
        if label.max()==1:
            label = (label).astype(np.uint8).flatten()
        elif label.max()==255:
            label = (label).astype(np.uint8).flatten() / 255
            #
            # label[label <=128] = 0
            # label[label>128]  = 1
            # label.astype('float')
            # label = label.flatten()
        else:
            raise RuntimeError('Please check your label.')
        # pdb.set_trace()

        # check the pixel value
        # pdb.set_trace()

        assert label.max() == 1 and (pred).max() <= 1
        assert label.min() == 0 and (pred).min() >= 0


        # test another datasets ISBI 2012
        # if cfg['DATASET'] == "ISBI2012":
        #     label = 1 - label
        #     pred = 1 - pred


        y_scores, y_true = pred, label

        # Area under the ROC curve
        # pdb.set_trace()
        fpr, tpr, thresholds = roc_curve((y_true), y_scores)
        AUC_ROC = roc_auc_score(y_true, y_scores)
        # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
        # print ("\nArea under the ROC curve: " +str(AUC_ROC))

        # ap_score = average_precision_score(y_true, y_scores)
        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
        recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
        AUC_prec_rec = np.trapz(precision, recall)
        # print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))

        # Confusion matrix
        threshold_confusion = threshold_confusion
        # print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
        y_pred = np.empty((y_scores.shape[0]))
        for i in range(y_scores.shape[0]):
            if y_scores[i] >= threshold_confusion:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        confusion = confusion_matrix(y_true, y_pred)
        # print (confusion)
        accuracy = 0
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        # print ("Global Accuracy: " +str(accuracy))
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        # print ("Specificity: " +str(specificity))
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        # print ("Sensitivity: " +str(sensitivity))
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        # print ("Precision: " +str(precision))

        # Jaccard similarity index
        # jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
        # print ("\nJaccard similarity score: " +str(jaccard_index))

        # F1 score
        F1_score = f1_score(y_true, y_pred, average='binary')
        # print ("\nF1 score (F-measure): " +str(F1_score))
        # print(1)



        # print(classification_report(label, pred, target_names=["class 0", "class 1"]))
        f1m.append(F1_score)
        accm.append(accuracy)
        aucm.append(AUC_ROC)
        specificitym.append(specificity)
        precisionm.append(precision)
        sensitivitym.append(sensitivity)

    # print("Your score of new data is {}".format(np.array(f1m).mean()))
    return np.array(f1m).mean(), np.array(accm).mean(), np.array(aucm).mean(), np.array(specificitym).mean(), np.array(
        precisionm).mean(), np.array(sensitivitym).mean()
class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
import configparser
import h5py
import sys



if __name__ == "__main__":
    threshold_confusion = 0.4


    best_h5 = []
    config = configparser.ConfigParser()
    config.read('configuration.txt')
    # config.read('configuration_STARE.txt')
    # config.read('configuration_CHASE.txt')
    name_experiment = config.get('experiment name', 'name')
    dataset = config.get('data attributes', 'dataset')
    path_experiment = './log/experiments/' + name_experiment + '/' + dataset + '/'


    log_path_experiment = './log/experiments/' + name_experiment + '/' + dataset + '/'
    TMP_DIR = log_path_experiment
    h5py_lists = natsort.natsorted(glob(os.path.join(path_experiment, '*.h5')),reverse=True)
    log = Logger(os.path.join(TMP_DIR, name_experiment +'-'+str(threshold_confusion) +'-eval-log.txt'))
    sys.stdout = log
    h5_file_num = 1
    h5py_path_one = h5py_lists[h5_file_num]
    print(h5py_path_one,'threshold_confusion is:',threshold_confusion)
    pre_image_data = h5py.File(h5py_path_one)
    y_gt = pre_image_data['y_gt']
    y_pred = pre_image_data['y_pred'][:]
    # f1, acc, auc, specificity, precision, sensitivity = metrics(y_gt, y_pred, threshold_confusion)
    evaluating_indicator = metrics(y_gt, y_pred, threshold_confusion)
    print("f1", evaluating_indicator[0], "accuracy", evaluating_indicator[1], "auc", evaluating_indicator[2],
          "specificity", evaluating_indicator[3], "precision", evaluating_indicator[4],
          "sensitivity", evaluating_indicator[5])

    print("*" * 100)
    print('*'*40)
    print('finish......')

