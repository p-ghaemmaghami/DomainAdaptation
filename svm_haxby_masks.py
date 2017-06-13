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
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

### Load subjects datasets ########################################################
source_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\'

subjects = (['subj1','subj2','subj3','subj4','subj5','subj6'])
subjects = (['subj1','subj2','subj3','subj4','subj6'])
masks = (['subj1','subj2','subj3','subj4','subj5','subj6','all'])
masks = (['subj1','subj2','subj3','subj4','subj5','subj6'])
masks = (['atlas'])


cv_scores=np.zeros([len(subjects),len(masks),12])
total_confmat=np.zeros([7,7])

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
        
        mask_filename = source + str(subjects[subs]) + str('_') + str(masks[mask]) + str('_mask4_vt.nii.gz')
        #mask_filename = source + str('VTC_adapted_native.nii.gz')
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
            
            #pca = PCA(n_components=1000)
            #pca.fit(x_train)
            #x_train = pca.transform(x_train)
            #x_test = pca.transform(x_test)
            
            model = OneVsRestClassifier(LinearSVC(random_state=0))
            parameters = {'estimator__C':[0.01,0.1,1,10]}
            clf = grid_search.GridSearchCV(model, parameters, score_func=accuracy_score)         
            
            clf.fit(x_train, y_train)
            prediction = clf.predict(x_test)
            cv_scores[subs,mask,run] = np.sum(prediction == y_test) / float(np.size(y_test))
            
            cm = confusion_matrix(y_test, prediction)
            total_confmat = total_confmat + cm
        
        classification_accuracy = np.mean(cv_scores[subs,mask,:])
        print(classification_accuracy)            
            
            
        target_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\all_haxby_masks.mat'
        sio.savemat(target_folder, {'cv_scores':cv_scores})



dict = sio.loadmat(target_folder)
cv_scores = dict['cv_scores']

sub=4
for i in range(6):
    np.mean(cv_scores[sub,i,range(11)])
    
for i in range(6):
    np.std(cv_scores[sub,i,range(11)])
    


target_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\confmat_orig.mat'
sio.savemat(target_folder, {'total_confmat':total_confmat})

total_confmat_normalized = total_confmat.astype('float') / total_confmat.sum(axis=1)[:, np.newaxis]
labels=["Bottle", "Cat", "Chair", "Face", "House", "Scissor", "Shoe"]
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(total_confmat_normalized)
#plt.title('Confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted Labels')
plt.ylabel('Ground-Truth Labels')
for i in range(np.size(total_confmat_normalized,0)):
    for j in range(np.size(total_confmat_normalized,1)):
        plt.text(j-.2, i+.2, float("{0:.2f}".format(total_confmat_normalized[i,j])), fontsize=14)
plt.show()


dict = sio.loadmat(target_folder)
total_confmat = dict['total_confmat']