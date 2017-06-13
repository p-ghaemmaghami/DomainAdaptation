# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:13:50 2015

@author: Pouya
"""

import numpy as np
import scipy.io as sio
from nilearn.input_data import NiftiMasker
import pandas
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import sklearn.preprocessing
from sklearn import grid_search
from sklearn.metrics import accuracy_score 

import matplotlib.pyplot as plt

### Load subjects datasets ########################################################
source_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\'

subjects = (['subj1','subj2','subj3','subj4','subj5','subj6'])
masks = (['thr0','thr10','thr20','thr30','thr40','thr50','thr60','thr70','thr80'])#,'thr90'

cv_scores=np.zeros([len(subjects),len(masks),12])

for subs in range(len(subjects)):
    print(subjects[subs])
    
    for mask in range(len(masks)):
        print(masks[mask])

        source = source_folder + str(subjects[subs]) + str('\\')
        
        labels = np.recfromcsv(source +str('labels.txt'), delimiter=" ")
        target = labels['labels']
        
        #words = set(['face','cat','bottle','house','scissors','shoe','scrambledpix','chair'])
        words = set(['face','cat','bottle','house','scissors','shoe','chair'])
        condition_mask = [ w in words for w in target ]
        condition_mask = np.array(condition_mask, dtype=bool).T
        target = target[condition_mask]
        
        target_int = pandas.get_dummies(target)
        target_int = target_int.values.argmax(1)   
        
        mask_filename = source + str('VTC_adapted_') + str(masks[mask]) + str('.nii_native.nii.gz')
        nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=False , detrend=True)
        func_filename = source + str('bold.nii.gz')
        fmri_masked = nifti_masker.fit_transform(func_filename)
        fmri_masked = fmri_masked[condition_mask]
            
        #fmri_masked = fmri_masked[target_int!=6,:]
        #target_int = target_int[target_int!=6]
        #target_int[target_int==7]=6
        
        #zero=np.all(fmri_masked!=0,axis=0)
        fmri_masked = fmri_masked[:,np.all(fmri_masked!=0,axis=0)]
            
        run_range = range(12)
        feat_range = range(756)
        if subs == 4:
            run_range = range(11)
            feat_range = range(693)
     
        for run in run_range:
            test_index = range(run*63,(run+1)*63)
            train_index = np.setdiff1d(feat_range,test_index)            
            #train_index = np.setdiff1d(range(693),test_index)
            
            x_test = fmri_masked[test_index,:]
            y_test = target_int[test_index]
            
            x_train = fmri_masked[train_index,:]
            y_train = target_int[train_index]
                
            # Normalize pixel intensities
            scaler = sklearn.preprocessing.StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            
            model = OneVsRestClassifier(LinearSVC(random_state=0))
            parameters = {'estimator__C':[0.01,0.1,1,10]}
            clf = grid_search.GridSearchCV(model, parameters, score_func=accuracy_score)         
            
            clf.fit(x_train, y_train)
            prediction = clf.predict(x_test)
            cv_scores[subs,mask,run] = np.sum(prediction == y_test) / float(np.size(y_test))
        
        classification_accuracy = np.mean(cv_scores[subs,mask,run_range])
        print(classification_accuracy)            
            
            
        target_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\all_vtc(atlas)_masks.mat'
        sio.savemat(target_folder, {'cv_scores':cv_scores})

#####################
target_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\all_vtc(atlas)_masks.mat'
dict = sio.loadmat(target_folder)
cv_scores = dict['cv_scores']

np.mean(cv_scores[0,0,range(12)])
np.mean(cv_scores[1,0,range(12)])
np.mean(cv_scores[2,0,range(12)])
np.mean(cv_scores[3,0,range(12)])
np.mean(cv_scores[4,0,range(11)])
np.mean(cv_scores[5,0,range(12)])

np.std(cv_scores[0,0,range(12)])
np.std(cv_scores[1,0,range(12)])
np.std(cv_scores[2,0,range(12)])
np.std(cv_scores[3,0,range(12)])
np.std(cv_scores[4,0,range(11)])
np.std(cv_scores[5,0,range(12)])

np.mean(np.concatenate((cv_scores[0,0,range(12)],cv_scores[1,0,range(12)],cv_scores[2,0,range(12)],cv_scores[3,0,range(12)],cv_scores[4,0,range(11)],cv_scores[5,0,range(12)]),0))

np.std(np.concatenate((cv_scores[0,0,range(12)],cv_scores[1,0,range(12)],cv_scores[2,0,range(12)],cv_scores[3,0,range(12)],cv_scores[4,0,range(11)],cv_scores[5,0,range(12)]),0))

#################### average across subject for different atlas thresholds
np.mean(np.concatenate((cv_scores[0,0,range(12)],cv_scores[1,0,range(12)],cv_scores[2,0,range(12)],cv_scores[3,0,range(12)],cv_scores[4,0,range(11)],cv_scores[5,0,range(12)]),0))
np.mean(np.concatenate((cv_scores[0,1,range(12)],cv_scores[1,1,range(12)],cv_scores[2,1,range(12)],cv_scores[3,1,range(12)],cv_scores[4,1,range(11)],cv_scores[5,1,range(12)]),0))
np.mean(np.concatenate((cv_scores[0,2,range(12)],cv_scores[1,2,range(12)],cv_scores[2,2,range(12)],cv_scores[3,2,range(12)],cv_scores[4,2,range(11)],cv_scores[5,2,range(12)]),0))
np.mean(np.concatenate((cv_scores[0,3,range(12)],cv_scores[1,3,range(12)],cv_scores[2,3,range(12)],cv_scores[3,3,range(12)],cv_scores[4,3,range(11)],cv_scores[5,3,range(12)]),0))
np.mean(np.concatenate((cv_scores[0,4,range(12)],cv_scores[1,4,range(12)],cv_scores[2,4,range(12)],cv_scores[3,4,range(12)],cv_scores[4,4,range(11)],cv_scores[5,4,range(12)]),0))
np.mean(np.concatenate((cv_scores[0,5,range(12)],cv_scores[1,5,range(12)],cv_scores[2,5,range(12)],cv_scores[3,5,range(12)],cv_scores[4,5,range(11)],cv_scores[5,5,range(12)]),0))
np.mean(np.concatenate((cv_scores[0,6,range(12)],cv_scores[1,6,range(12)],cv_scores[2,6,range(12)],cv_scores[3,6,range(12)],cv_scores[4,6,range(11)],cv_scores[5,6,range(12)]),0))
np.mean(np.concatenate((cv_scores[0,7,range(12)],cv_scores[1,7,range(12)],cv_scores[2,7,range(12)],cv_scores[3,7,range(12)],cv_scores[4,7,range(11)],cv_scores[5,7,range(12)]),0))
np.mean(np.concatenate((cv_scores[0,8,range(12)],cv_scores[1,8,range(12)],cv_scores[2,8,range(12)],cv_scores[3,8,range(12)],cv_scores[4,8,range(11)],cv_scores[5,8,range(12)]),0))

np.std(np.concatenate((cv_scores[0,0,range(12)],cv_scores[1,0,range(12)],cv_scores[2,0,range(12)],cv_scores[3,0,range(12)],cv_scores[4,0,range(11)],cv_scores[5,0,range(12)]),0))
np.std(np.concatenate((cv_scores[0,1,range(12)],cv_scores[1,1,range(12)],cv_scores[2,1,range(12)],cv_scores[3,1,range(12)],cv_scores[4,1,range(11)],cv_scores[5,1,range(12)]),0))
np.std(np.concatenate((cv_scores[0,2,range(12)],cv_scores[1,2,range(12)],cv_scores[2,2,range(12)],cv_scores[3,2,range(12)],cv_scores[4,2,range(11)],cv_scores[5,2,range(12)]),0))
np.std(np.concatenate((cv_scores[0,3,range(12)],cv_scores[1,3,range(12)],cv_scores[2,3,range(12)],cv_scores[3,3,range(12)],cv_scores[4,3,range(11)],cv_scores[5,3,range(12)]),0))
np.std(np.concatenate((cv_scores[0,4,range(12)],cv_scores[1,4,range(12)],cv_scores[2,4,range(12)],cv_scores[3,4,range(12)],cv_scores[4,4,range(11)],cv_scores[5,4,range(12)]),0))
np.std(np.concatenate((cv_scores[0,5,range(12)],cv_scores[1,5,range(12)],cv_scores[2,5,range(12)],cv_scores[3,5,range(12)],cv_scores[4,5,range(11)],cv_scores[5,5,range(12)]),0))
np.std(np.concatenate((cv_scores[0,6,range(12)],cv_scores[1,6,range(12)],cv_scores[2,6,range(12)],cv_scores[3,6,range(12)],cv_scores[4,6,range(11)],cv_scores[5,6,range(12)]),0))
np.std(np.concatenate((cv_scores[0,7,range(12)],cv_scores[1,7,range(12)],cv_scores[2,7,range(12)],cv_scores[3,7,range(12)],cv_scores[4,7,range(11)],cv_scores[5,7,range(12)]),0))
np.std(np.concatenate((cv_scores[0,8,range(12)],cv_scores[1,8,range(12)],cv_scores[2,8,range(12)],cv_scores[3,8,range(12)],cv_scores[4,8,range(11)],cv_scores[5,8,range(12)]),0))


sub = np.zeros([6,2,9])
for i in range(6):
    run_range = range(12)
    if i == 4:
        run_range = range(11)
    for j in range(9):
        sub[i,0,j] = np.mean(cv_scores[i,j,run_range])
        sub[i,1,j] = np.std(cv_scores[i,j,run_range])
        
np.mean(np.squeeze(sub[:,0,:]),0)
    


# example data
x = np.arange(0, 9, 1)
y = np.exp(-x)
ls = 'dotted'

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# standard error bars
plt.errorbar(x, sub[0,0,:], marker='o', ms=8, xerr=None, yerr=sub[0,1,:], ls=ls, color='black', label="test1")
plt.errorbar(x, sub[1,0,:], marker='o', ms=8, xerr=None, yerr=sub[1,1,:], ls=ls, color='blue')
plt.errorbar(x, sub[2,0,:], marker='o', ms=8, xerr=None, yerr=sub[2,1,:], ls=ls, color='green')
plt.errorbar(x, sub[3,0,:], marker='o', ms=8, xerr=None, yerr=sub[3,1,:], ls=ls, color='red')
plt.errorbar(x, sub[4,0,:], marker='o', ms=8, xerr=None, yerr=sub[4,1,:], ls=ls, color='magenta')
plt.errorbar(x, sub[5,0,:], marker='o', ms=8, xerr=None, yerr=sub[5,1,:], ls=ls, color='cyan')

plt.xlim((-0.5, 8.5))
plt.xticks(x,('thr=0','thr=0.1','thr=0.2','thr=0.3','thr=0.4','thr=0.5','thr=0.6','thr=0.7','thr=0.8'))
plt.legend(['Sub1','Sub2','Sub3','Sub4','Sub5','Sub6'], loc= 'lower left', fontsize='x-small')
#plt.legend(['Sub1','Sub2','Sub3','Sub4','Sub5','Sub6'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
