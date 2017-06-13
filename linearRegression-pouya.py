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

import deeppy as dp

#### Parameters ######################
batch_size = 50 # 10, 50, 100
neuronNum = [100,8] # [1000,500,100,8] [500,300,100,15] [500,300,100,4] [300,200,100,4]
corruptionRatio =  0.1  #  0.01, 0.1, "0.25"
epochs_value_pretraining = [50] # 50
epochs_value_finetune = [50] # 50
learning_rate_pretraining = 0.05 # 0.01, 0.05, 0.1
learning_rate_finetune = 0.05 # 0.01, 0.05, 0.1
momentum_value = 0.9

source = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\imagenet_pca10(acc=91)_cnn_feat.mat'
dict = sio.loadmat(source)
feat = dict['pca_feat']


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
        
        # no augment
        test_index = range(run*72,(run+1)*72)
        train_index = np.setdiff1d(range(864),test_index)
    
        x_test = features[test_index,:]
        y_test = targets[0,test_index]
        y_test[y_test==7]=6
        x_test[y_test==6,:] = np.NAN;
        y_test[y_test==6,] = np.NAN;
        x_test = x_test[~np.all(np.isnan(x_test),axis=1)]
        y_test = y_test[~np.isnan(y_test)]
        y_test = y_test.astype('int')
        y_test[y_test==7]=6
        
        x_train = features[train_index,:]
        y_train = targets[0,train_index]
        x_train[y_train==6,:] =np.NAN;
        y_train[y_train==6,] = np.NAN;
        x_train = x_train[~np.all(np.isnan(x_train),axis=1)]
        y_train = y_train[~np.isnan(y_train)]
        y_train[y_train==7]=6
        y_train = y_train.astype('int')
                
        # Normalize features
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
     
        # cnn represenations       
        cnn_rep = np.concatenate((feat[0,0],feat[0,1],feat[0,2],feat[0,3],feat[0,4],feat[0,5],feat[0,6]),0)
        cnn_targets = np.concatenate((0*np.ones(len(feat[0,0])), 1*np.ones(len(feat[0,1])), 2*np.ones(len(feat[0,2])),3*np.ones(len(feat[0,3])), 4*np.ones(len(feat[0,4])), 5*np.ones(len(feat[0,5])),6*np.ones(len(feat[0,6]))),0)
        
        scaler = preprocessing.StandardScaler()
        cnn_rep = scaler.fit_transform(cnn_rep)
        
        # brain representations
        brain_rep = np.zeros([len(cnn_targets),np.size(x_train,1)])
        for i in range(len(cnn_targets)):
            selectedTarget = cnn_targets[i]
            selectedTrainset = x_train[y_train==selectedTarget,:]

            selectedTrainID = np.random.randint(0,np.size(selectedTrainset,0))
            brain_rep[i,:] = selectedTrainset[selectedTrainID,:]
        
        # Linear Regression
        numberOfModels = 10
        model = np.zeros([numberOfModels,],dtype='object')
        for i in range(numberOfModels):
            model[i] = linear_model.LinearRegression()
            model[i].fit(brain_rep, cnn_rep[:,i])
            print(i)
            
        # Prediction
        pred = np.zeros([x_test.shape[0],numberOfModels])
        pred_train = np.zeros([x_train.shape[0],numberOfModels])
        for i in range(numberOfModels):
            pred[:,i] = model[i].predict(x_test)
            pred_train[:,i] =  model[i].predict(brain_rep)
        
        cor = np.zeros([numberOfModels,])
        for i in range(numberOfModels):
            cor[i] = np.corrcoef(cnn_rep[:,i],pred_train[:,i])[0,1]
        
        
        svm_model = OneVsRestClassifier(LinearSVC(random_state=0))      
        svm_model.fit(cnn_rep[:,0:numberOfModels], cnn_targets)
        prediction = svm_model.predict(pred)
        
        acc = np.sum(prediction == y_test) / float(np.size(y_test))
        
        
        
        
        
        # Setup neural network using the stacked autoencoder layers
        net = dp.NeuralNetwork(
            [
                dp.FullyConnected(
                    n_out=100,#neuronNum[-1],
                    weights=dp.Parameter(dp.AutoFiller()),
                ),
                dp.Sigmoid(),
                dp.FullyConnected(
                    n_out=10,
                    weights=dp.Parameter(dp.AutoFiller()),
                ),
            ],
            loss=dp.loss.MeanSquaredError(),
        )

        # Fine-tune neural network
        train_input = dp.SupervisedInput(brain_rep, cnn_rep, batch_size=batch_size)
        test_input = dp.Input(x_test)
        for i in range(len(epochs_value_pretraining)):
            lr = learning_rate_pretraining/10**i
            trainer = dp.StochasticGradientDescent(
                max_epochs=epochs_value_pretraining[i], learn_rule=dp.Momentum(learn_rate=lr, momentum=momentum_value),
            )
            trainer.train(net, train_input)
        
       
        net = dp.NeuralNetwork(
        layers=[
            net.layers[0],
            ],
            loss=dp.SoftmaxCrossEntropy(),
        )
        
        net2 = dp.NeuralNetwork(
            [
                dp.FullyConnected(
                    n_out=8,#neuronNum[-1],
        #            n_out=dataset.n_classes,
                    weights=dp.Parameter(dp.AutoFiller()),
                ), 
            ],
            loss=dp.SoftmaxCrossEntropy(),
        )        
        
        train_input = dp.SupervisedInput(x_train, y_train, batch_size=batch_size)
        test_input = dp.Input(x_test)
        for i in range(len(epochs_value_finetune)):
            lr = learning_rate_finetune/10**i
            trainer = dp.StochasticGradientDescent(
                max_epochs=epochs_value_finetune[i], learn_rule=dp.Momentum(learn_rate=lr, momentum=momentum_value),
            )
            trainer.train(net2, train_input)
        
        
        # Evaluate on test data
        acc = np.mean(net2.predict(test_input) == y_test)
        print('Test acc rate: %.4f' % acc)

        
        
        
        
       
        cv_scores[run,subs] = acc
        
    classification_accuracy = np.mean(cv_scores[:,subs])
    print(classification_accuracy)
    
    target_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\Augmented_oneBlockOut_NN.mat'
    sio.savemat(target_folder, {'cv_scores':cv_scores, 'batch_size':batch_size, 'neuronNum':neuronNum, 'corruptionRatio':corruptionRatio, 'learning_rate_overall':learning_rate_overall, 'momentum_value':momentum_value})
    
#dict = sio.loadmat(target_folder)
#cv_scores = dict['cv_scores']
