# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:15:11 2015

@author: Pouya
"""

import deeppy as dp
import numpy as np
import scipy.io as sio
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import sklearn.preprocessing
from sklearn import grid_search
from sklearn.metrics import accuracy_score 


### Load subjects datasets ########################################################
source_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\'

subjects = (['subj1','subj2','subj3','subj4','subj6'])
cv_scores=np.zeros([12,5])

for subs in range(len(subjects)):
    
    print(subjects[subs])
    folder = 'c:\\Users\\Pouya\\Documents\\haxby\\'+str(subjects[subs])+'\\_uniform_augmentedData_new.mat'
    
    dict = sio.loadmat(folder)
    features = dict['features']
    targets = dict['targets']
    #n_samples = dict['n_samples']

    
    for run in range(12):
        test_index = range(run*72,(run+1)*72)
        
        x_test = features[test_index,:]
        y_test = targets[0,test_index]
        y_test = y_test.astype('int')
        
        numberOfAugmentation = 100   
        for i in range(numberOfAugmentation):
            for j in range(len(test_index)):
                features [i*864+test_index[j],:] = np.NAN
                targets [0,i*864+test_index[j]] = np.NAN
                
        x_train = np.delete(features, test_index, axis=0)
        y_train = np.delete(targets, test_index)
                
        x_train = features[~np.all(np.isnan(features),axis=1)]
        y_train = targets[~np.isnan(targets)]
        y_train = y_train.astype('int')
        
        
        
        # Normalize pixel intensities
        scaler = dp.UniformScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        # Shufflinig
        ind = range(len(y_train))
        np.random.shuffle(ind)
        y_train=y_train[ind]
        x_train=x_train[ind,:]
        
        # Prepare autoencoder input
        batch_size = 50
        train_input = dp.Input(x_train, batch_size=batch_size)
        
        # Setup autoencoders
        sae = dp.StackedAutoencoder(
            layers=[
                dp.DenoisingAutoencoder(
                    n_out=300,
                    weights=dp.Parameter(dp.AutoFiller()),
                    activation='sigmoid',
                    corruption=0.1,
                ),
                dp.DenoisingAutoencoder(
                    n_out=100,
                    weights=dp.Parameter(dp.AutoFiller()),
                    activation='sigmoid',
                    corruption=0.1,
                ),
                
            ],
        )
        
        # Train autoencoders layer-wise
        trainer = dp.StochasticGradientDescent(
            max_epochs=15, learn_rule=dp.Momentum(learn_rate=0.05, momentum=0.9),
        )
        
        for ae in sae.ae_models():
            trainer.train(ae, train_input)
        
        # Train stacked autoencoders
        trainer.train(sae, train_input)
        
        # Setup neural network using the stacked autoencoder layers
        net = dp.NeuralNetwork(
            layers=sae.feedforward_layers() + [
                dp.FullyConnected(
                    n_out=8,
        #            n_out=dataset.n_classes,
                    weights=dp.Parameter(dp.AutoFiller()),
                ),
            ],
            loss=dp.SoftmaxCrossEntropy(),
        )
        
        # Fine-tune neural network
        train_input = dp.SupervisedInput(x_train, y_train, batch_size=batch_size)
        test_input = dp.Input(x_test)
        trainer = dp.StochasticGradientDescent(
            max_epochs=50, learn_rule=dp.Momentum(learn_rate=0.05, momentum=0.9),
        )
        trainer.train(net, train_input)
        
        # Evaluate on test data
        error = np.mean(net.predict(test_input) != y_test)
        print('Test error rate: %.4f' % error)

        
        print(cv_scores[run,subs])
        
    classification_accuracy = np.mean(cv_scores[:,subs])
    print(classification_accuracy)
    










    
target_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\Augmented_oneBlockOut_SVM.mat'
sio.savemat(target_folder, {'cv_scores':cv_scores})