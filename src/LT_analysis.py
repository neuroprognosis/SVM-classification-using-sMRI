#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:43:00 2022

@author: agupta
"""

#import necessary libraries
import sklearn
import os
import time

import re
import numpy as np
import pandas as pd

from scipy.io import loadmat
import nibabel as nib
from nilearn.input_data import MultiNiftiMasker

from sklearn.svm import SVC  
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import make_scorer,roc_auc_score,f1_score

from collections import Counter
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc,accuracy_score
from scipy import interp 
from numpy import sqrt, argmax
from sklearn.utils import shuffle

"""
Compute run time
"""
def tic():
    global _start_time 
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))


"""
Read image data 
Keyword Args:
    path: path to image directory
    regex_match: read specific tissue segment images (p1*gray, p2*white, p3*csf)
    mask: path to predefined mask
    smooth: value for gaussian (FWHM) smoothing intensity
"""
def readDataRI(path, regex_match, mask, smooth):
    subjects_id = []
    list_files  = []
    masker = MultiNiftiMasker(mask_img=mask, n_jobs=-1, smoothing_fwhm=smooth)
    for files in os.listdir(path):
        if re.match(regex_match, files):
            list_files.append(path+files)
            sub_id = files[-13:-4]
            subjects_id.append(sub_id)
    list_files = np.sort(list_files)
    subjects_id = np.sort(subjects_id).tolist()
    list_files = list_files.tolist()
    data  = masker.fit_transform(list_files)
    data = np.vstack(data)
    return data, subjects_id



"""
Read DELCODE data and store in a dataframe
Keyword Args:
    path: path to DELCODE file
"""
def readMatData(mat_path):
    data = loadmat(mat_path, squeeze_me=True)
    demo = data['Demo']
    body = data['Body']
    csf = data['Csf']
    cog = data['Cog']
    vol = data['Vol']
    cog = cog['X'][()]
    csf = csf['X'][()]
    hippo = vol['hipp'][()]['X'][()]
    delcode_data = pd.DataFrame()
    delcode_data['subj_id'] = data['IDs'][()]
    delcode_data['id'] = demo['IDn'][()]
    delcode_data['age'] = demo['age_bl'][()]
    delcode_data['bmi'] = body['bmi'][()]
    delcode_data['diag'] = demo['diag_bl'][()]
    delcode_data['sex'] = demo['sex'][()]
    delcode_data['edu'] = demo['edu'][()]
    delcode_data['fmem'] = cog[:, 21]
    delcode_data['ptau'] = csf[:, 4]
    delcode_data['abeta4240'] = csf[:, 5]
    return delcode_data

 
path_lr = '/storage/users/agupta/SPM_CATr1888avg_Shoot_DCTemplaterp123_1p0mm_mwmp1avg/longit_3scans_rates_alpha1e3_s6/'
mask = nib.load('/storage/users/agupta/mask/icvmask.nii')
mat = readMatData('/storage/users/agupta/delcode_cov1079.mat')


_start_time = time.time()
tic()
data_lr, subjects_id_lr = readDataRI(path_lr, 's6mwmp1*', mask, 9)
tac()

# create a dataframe and store image data and map it to subject IDs
subjects_lr = pd.DataFrame()
subjects_lr['subjs'] = np.array(subjects_id_lr)
subjects_lr['diag'] = subjects_lr.subjs.map(mat.set_index('subj_id')['diag'])
subjects_lr = subjects_lr.dropna() 
data_lr = data_lr[subjects_lr.index]
label_lr = np.array(subjects_lr.diag, dtype = 'int64') 


# data filtering for binary classification 
data_lr_f = data_lr[subjects_lr.diag.isin([0,2])]
label_lr_f = label_lr[subjects_lr.diag.isin([0,2])]

# assign binary labels to filtered data
label_bin = []
for val in label_lr_f:
    if (val == 2):
        label_bin.append(1)
    elif (val == 0):
        label_bin.append(0)        
label_bin = np.array(label_bin)

# splitting data into training and test set while maintatining the class balance 
X_train, X_test, y_train, y_test = train_test_split(data_lr_f, label_bin, test_size=0.33, stratify=label_bin)


#data filtering for multivariate classifier
data_lr_f = data_lr[subjects_lr.diag.isin([0,1,2])]
label_lr_f = label_lr[subjects_lr.diag.isin([0,1,2])]

# splitting data into training and test set while maintatining the class balance 
X_train, X_test, y_train, y_test = train_test_split(data_lr_f, label_lr_f, test_size=0.33, stratify=label_lr_f)

#binarize the class labels 
label_bin_m = label_binarize(label_lr_f, classes=[0, 1, 2]) 
X_train, X_test, y_train, y_test = train_test_split(data_lr_f, label_bin_m, test_size=0.33, stratify=label_bin_m)


"""
Draw a Cross Validated ROC Curve.
Keyword Args:
    classifier: Classifier Object
    cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
    X: Feature Pandas DataFrame
    y: Response Numpy array
    title: customize title for ROC Curve
Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
"""
def draw_cv_roc_curve(classifier, cv, X, y, title='ROC Curve'):
    
    # Creating ROC Curve with Cross Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC curve of fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
        
    #plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='black',
    #         label='Guess', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC curve(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.grid()
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()
    
 
    
"""
Draw ROC Curves for multiclass classification.
Keyword Args:
        classifier: Classifier Object
        X: Feature Pandas DataFrame
        y: Response Numpy array
Example largely taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
"""
def draw_multi_roc_curve(classifier, X, y, title='ROC Curve'):

    #n_classes = y.shape[1]
    #print(n_classes)
    y_score = classifier.fit(X, y).decision_function(X_test)
    
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lw=1
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # Compute micro-average ROC curve and ROC area
    fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    colors = cycle(['orange','turquoise','indianred'])
        
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))
        
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr['macro']=all_fpr
    tpr['macro']=mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
       
    plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="Average ROC (AUC = {0:0.2f})".format(roc_auc["macro"]),
    color="blue",
    linestyle="solid",
    linewidth=2,
    )
    
    plt.grid()
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('')#ROC for {}'.format(item["name"])
    plt.legend()
    #plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
    plt.show()  
    
    
    
    
comparison_items_lr = [{
    "name": "standard_SVM",
    "model": None,
    "X": X_train,
    "y": y_train
},{
    "name": "weighted_SVM",
    "model": None,
    "X": X_train,
    "y": y_train
}]

   
###################################################################
#Binary Classification
n_classes=2

# CV Scores and ROC plots for binary Classification 
for item in comparison_items_lr:
    print(item["name"])
    print(Counter(item["y"])) 
    
    # define parameter grid
    C = [50, 10, 1.0, 0.1, 0.01]
    grid = dict(C=C)  
    item["X"], item["y"] = shuffle(item["X"], item["y"]) 
    
    # define cross-validation scheme
    k=5
    skf = StratifiedKFold(n_splits=k,
                                       shuffle=True)
    
    # define scoring scheme
    scoring = {'accuracy': make_scorer(accuracy_score),
               'F1_score': make_scorer(f1_score, average = 'macro')}
    
    if(item["name"]=='weighted_SVM'):
        print('Weighted SVM')
        clf = SVC(kernel='linear', probability=True, class_weight = 'balanced')
    else:
      print('Standard SVM')
      clf = SVC(kernel='linear',probability=True)  
    
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, cv=skf)
    
    # calculating defined scores by cross-validation, also records fit/score time
    cv_score = model_selection.cross_validate(clf, item["X"], item["y"], cv=skf, scoring=scoring, n_jobs=-1)
    item["cv_score"] = cv_score
    results = {k:sum(x)/len(x) for k,x in item["cv_score"].items()}
    print("Average Accuracy on k folds: ", results['test_accuracy'])
    print("Average F1 Score on k folds: ", results['test_F1_score'])
    

# test scores for binary Classification
for item in comparison_items_lr:
    print(item["name"],Counter(y_test))
    k=5
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    
    if(item["name"]=='weighted_SVM'):
      print('Weighted SVM')
      clf = SVC(kernel='linear', probability=True, class_weight = 'balanced')
    else:
      print('Standard SVM')
      clf = SVC(kernel='linear',probability=True) 
      
    classifier = clf.fit(item["X"], item["y"])
    y_pred = classifier.predict(X_test)
    #calculate test scores
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    test_acc = accuracy_score(y_test, y_pred)
    item["test_f1_score"] = test_f1
    item["test_acc"] = test_acc
    print('Test Score_f1: ', item["test_f1_score"])
    print('Test Score_acc: ', item["test_acc"])
    
    
# test ROC plots for binary classification
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc', 'x_o', 'y_o', 'optThres'])

# plot ROC curves for binary classification  
for item in comparison_items_lr:    
    
    if(n_classes > 2):
      print('Multiclass Classification')
      clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    elif(item["name"]=='weighted_SVM'):
        print('Weighted SVM')
        clf = SVC(kernel='linear', probability=True, class_weight = 'balanced')
    else:
      print('Standard SVM')
      clf = SVC(kernel='linear',probability=True) 
    
    # generate a no skill prediction (majority class)
    ns_proba = [0 for _ in range(len(y_test))]
    clf.fit(item["X"], item["y"])
    
    # predict probabilities    
    y_proba = clf.predict_proba(X_test)[::,1]
    
    #y_proba = classifier.predict_proba(X_test)[::,1]
    
    fpr, tpr, thresholds = roc_curve(y_test,  y_proba)
    auc = roc_auc_score(y_test, y_proba)
    #calculate G-mean for each threshold
    gmeans = sqrt(tpr*(1-fpr))
    # locate the index of the largest mean
    ix= argmax(gmeans)
    #x_o, y_o=fpr[ix], tpr[ix]
    print('Optimum Threshold=%f, G-mean=%0.3f' % (thresholds[ix], gmeans[ix]))
    result_table = result_table.append({'classifiers':item["name"],
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc,
                                        'x_o':fpr[ix],
                                        'y_o':tpr[ix],
                                        'optThres':thresholds[ix]}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)


# plot the results
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}(AUC={:.2f})".format(i, result_table.loc[i]['auc']))
       
    #plt.plot(result_table.loc[i]['x_o'],result_table.loc[i]['y_o'], 
     #        marker = 'o', color='red', 
      #       label="Optimum Threshold={:.2f}".format(result_table.loc[i]['optThres']))
    
    #, marker = 'o', color='red', label='Optimum Threshold')
    
#plt.plot([0,1], [0,1], color='black', linestyle='--')
plt.grid()
plt.xticks(np.arange(0.0, 1.1, step=0.2))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.2))
plt.ylabel("True Positive Rate", fontsize=15)


plt.title('', fontweight='bold', fontsize=15)
plt.legend(loc='lower right', fontsize=15)

plt.show()
   

#################################################
# Multiclass Classification
n_classes=3

# calculate CV Scores for multiclass classification
for item in comparison_items_lr:
    print(item["name"])
    print(Counter(item["y"]))
    k=5
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    
    if(item["name"]=='weighted_SVM'):
        print('Weighted Classification')
        clf = SVC(kernel='linear', decision_function_shape='ovr', 
                  probability=True, class_weight = 'balanced')
    else:
       clf = SVC(kernel='linear', decision_function_shape='ovr', 
                 probability=True)
    
    # define scoring scheme
    scoring = {'accuracy': make_scorer(accuracy_score),
               'F1_score': make_scorer(f1_score, average = 'macro')}
    
    # calculating defined scores by cross validation, also records fit/score time
    cv_score = model_selection.cross_validate(clf, item["X"], item["y"], cv=skf, scoring=scoring, n_jobs=-1)
    item["cv_score"] = cv_score
    results = {k:sum(x)/len(x) for k,x in item["cv_score"].items()}
    print("Average Accuracy on k folds: ", results['test_accuracy'])
    print("Average F1 Score on k folds: ", results['test_F1_score'])  
    
    
# calculate test scores for multiclass classification
for item in comparison_items_lr:
    print(item["name"],Counter(y_test))
    
    if(item["name"]=='weighted_SVM'):
       print('Weighted SVM')
       clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight = 'balanced'))
    else:
      print('Standard SVM')
      clf = OneVsRestClassifier(SVC(kernel='linear',probability=True)) 
      
    classifier = clf.fit(item["X"], item["y"])
    y_pred = classifier.predict(X_test)
    #calculating defined scores by cross validation, also records fit/score time
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    test_acc = accuracy_score(y_test, y_pred)
    item["test_f1_score"] = test_f1
    item["test_acc"] = test_acc    
    print('Test Score_f1: ', item["test_f1_score"])
    print('Test Score_acc: ', item["test_acc"])    


# plot ROC curves for multiclass classification     
for item in comparison_items_lr: 
    print(item["name"])
    #n_classes = item["y"].shape[1]
    if(item["name"]=='SVM_weighted'):
        print('Weighted Classification')
        clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight = 'balanced'))
    else:
        clf = OneVsRestClassifier(SVC(kernel='linear', probability=True)) 
    draw_multi_roc_curve(clf, item["X"], item["y"], title = '')
    


