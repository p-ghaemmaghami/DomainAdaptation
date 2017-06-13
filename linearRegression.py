# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:50:57 2015

@author: Pouya
"""

import numpy as np
import scipy.io as sio
from sklearn import preprocessing, linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

source = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\imagenet_pca_cnn_feat.mat'
dict = sio.loadmat(source)
feat = dict['pca_feat']

newFeat = np.concatenate((feat[0,0],feat[0,1],feat[0,2],feat[0,3],feat[0,4],feat[0,5],feat[0,6]),0)
newTargets = np.concatenate((0*np.ones(len(feat[0,0])), 1*np.ones(len(feat[0,1])), 2*np.ones(len(feat[0,2])), 3*np.ones(len(feat[0,3])), 4*np.ones(len(feat[0,4])), 5*np.ones(len(feat[0,5])), 6*np.ones(len(feat[0,6])) ),0)

#####################################

### Load subjects datasets ########################################################
source_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\'

subjects = (['subj1','subj2','subj3','subj4','subj6'])
cv_scores=np.zeros([12,5])

for subs in range(len(subjects)):
    
    print(subjects[subs])
    #n_samples = dict['n_samples']
    
    for run in range(12):

        folder = 'c:\\Users\\Pouya\\Documents\\haxby\\'+str(subjects[subs])+'\\_normal_augmentedData_new.mat'
        dict = sio.loadmat(folder)
        
        features = dict['features']
        targets = dict['targets']          
        
        test_index = range(run*72,(run+1)*72)
        
        x_test = features[test_index,:]
        y_test = targets[0,test_index]
        
        numberOfAugmentation = 100   
        for i in range(numberOfAugmentation):
            for j in range(len(test_index)):
                features [i*864+test_index[j],:] = np.NAN
                targets [0,i*864+test_index[j]] = np.NAN
                
        x_train = np.delete(features, test_index, axis=0)
        y_train = np.delete(targets, test_index)
                
        x_train = features[~np.all(np.isnan(features),axis=1)]
        y_train = targets[~np.isnan(targets)]
        
        x_train[y_train==6,:] =np.NAN;
        y_train[y_train==6,] = np.NAN;
        x_train = x_train[~np.all(np.isnan(x_train),axis=1)]
        y_train = y_train[~np.isnan(y_train)]
        y_train[y_train==7]=6
       
        y_train = y_train.astype('int')

        x_test[y_test==6,:] = np.NAN;
        y_test[y_test==6,] = np.NAN;
        x_test = x_test[~np.all(np.isnan(x_test),axis=1)]
        y_test = y_test[~np.isnan(y_test)]
        y_test = y_test.astype('int')
        y_test[y_test==7]=6
        
        
        # no augment
        test_index = range(run*72,(run+1)*72)
        train_index = np.setdiff1d(range(864),test_index)
    
        x_test = features[test_index,:]
        y_test = targets[0,test_index]
        y_test[y_test==7]=6

        
        x_train = features[train_index,:]
        y_train = targets[0,train_index]
        y_train[y_train==7]=6
                
        # Normalize features
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        
        # Preparing outputs
        outputs = np.zeros([len(y_train),feat[0,0].shape[1]])
        for i in range(len(y_train)):
            n = np.random.randint(0,feat[0,y_train[i]].shape[0])
            outputs[i,:] = feat[0,y_train[i]][n,:]
        
        scaler = preprocessing.StandardScaler()
        outputs = scaler.fit_transform(outputs)
        
        
        # augment based on output represenations
                
        
        # Linear Regression
        numberOfModels = 5
        model = np.zeros([numberOfModels,],dtype='object')
        for i in range(numberOfModels):
            model[i] = linear_model.LinearRegression()
            model[i].fit(x_train,outputs[:,i])
            print(i)
            
        # Prediction
        pred = np.zeros([x_test.shape[0],numberOfModels])
        pred_train = np.zeros([x_train.shape[0],numberOfModels])
        for i in range(numberOfModels):
            pred[:,i] = model[i].predict(x_test)
            pred_train[:,i] =  model[i].predict(x_train)
        
        cor = np.zeros([numberOfModels,])
        for i in range(numberOfModels):
            cor[i] = np.corrcoef(outputs[:,i],pred_train[:,i])[0,1]
        
        
        svm_model = OneVsRestClassifier(LinearSVC(random_state=0))      
        
        svm_model.fit(outputs[:,0:numberOfModels], y_train)
        prediction = svm_model.predict(pred)
        
        acc = np.sum(prediction == y_test) / float(np.size(y_test))
        
        
        
        
        
        
       
        cv_scores[run,subs] = acc
        
    classification_accuracy = np.mean(cv_scores[:,subs])
    print(classification_accuracy)
    
    target_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\Augmented_oneBlockOut_NN.mat'
    sio.savemat(target_folder, {'cv_scores':cv_scores, 'batch_size':batch_size, 'neuronNum':neuronNum, 'corruptionRatio':corruptionRatio, 'learning_rate_overall':learning_rate_overall, 'momentum_value':momentum_value})
    
#dict = sio.loadmat(target_folder)
#cv_scores = dict['cv_scores']
