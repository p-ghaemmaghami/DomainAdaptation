
import numpy as np
import scipy.io as sio
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import grid_search
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation

from sklearn.decomposition import SparseCoder
#from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import MiniBatchDictionaryLearning



#source = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\haxby_fc7_pca10_feat.mat'
#imagenet_targets = range(7)
#imagenet_targets = np.repeat(imagenet_targets,48)
#imagenet_features = feat
#source = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\haxby_cnn_feat.mat'
source = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\imagenet_cnn_feat.mat'
dict = sio.loadmat(source)
feat = dict['feat']
imagenet_features = np.concatenate((feat[0,0],feat[0,1],feat[0,2],feat[0,3],feat[0,4],feat[0,5],feat[0,6]))
imagenet_targets = np.concatenate((0*np.ones([feat[0,0].shape[0]]),1*np.ones([feat[0,1].shape[0]]),2*np.ones([feat[0,2].shape[0]]),3*np.ones([feat[0,3].shape[0]]),4*np.ones([feat[0,4].shape[0]]),5*np.ones([feat[0,5].shape[0]]),6*np.ones([feat[0,6].shape[0]]),))

#####################################

### Load subjects datasets ########################################################
source_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\'

subjects = (['subj1','subj2','subj3','subj4','subj5','subj6'])
subjects = (['subj1','subj2','subj3','subj4','subj6'])
cv_scores=np.zeros([12,5])

for subs in range(len(subjects)):
    
    print(subjects[subs])

    folder = 'c:\\Users\\Pouya\\Documents\\haxby\\'+str(subjects[subs])+'\\_normal_augmentedData_new.mat'
    
    folder = 'c:\\Users\\Pouya\\Documents\\haxby\\'+str(subjects[subs])+'\\_uniform_augmentedData_10_VTC_Native.mat'
    dict = sio.loadmat(folder)
    
    features = dict['features']
    
    # dimensionality scaling
    scaler = sklearn.preprocessing.StandardScaler()
    imagenet_features = scaler.fit_transform(imagenet_features)

    pca = PCA(n_components=np.size(features,1))
    pca.fit(imagenet_features)
    pca_feat = pca.transform(imagenet_features)
    
    # Dictionary Learning
    #dict_sparse = DictionaryLearning(alpha=1, n_components=150, max_iter=10, verbose=3)
    dict_sparse = MiniBatchDictionaryLearning(alpha=1, n_components=150, verbose=3)
    dict_sparse.fit(pca_feat)
    D = dict_sparse.components_
    
    #newFeat = dict_sparse.transform(pca_feat)
    
    #model = OneVsRestClassifier(LinearSVC(random_state=0))
    #parameters = {'estimator__C':[0.01,0.1,1,10]}
    #clf = grid_search.GridSearchCV(model, parameters, score_func=accuracy_score)     
    #scores = cross_validation.cross_val_score(clf, newFeat, imagenet_targets, cv=10)
    
    for run in range(12):
    
        #folder = 'c:\\Users\\Pouya\\Documents\\haxby\\'+str(subjects[subs])+'\\_normal_augmentedData_new.mat'
        #folder = 'c:\\Users\\Pouya\\Documents\\haxby\\'+str(subjects[subs])+'\\_normal_augmentedData_100_VTC_Native.mat'
        folder = 'c:\\Users\\Pouya\\Documents\\haxby\\'+str(subjects[subs])+'\\_uniform_augmentedData_10_VTC_Native.mat'
        dict = sio.loadmat(folder)
        
        features = dict['features']
        targets = dict['targets']
        targets = targets[0,:]
        
        features = features[0:864,:]
        targets = targets[0:864]
        
        #features = features[0:792,:]
        #targets = targets[0:792]
        #targets = targets.astype('float')
                
        
        test_index = range(run*72,(run+1)*72)
        
        x_test = features[test_index,:]
        y_test = targets[test_index]
                       
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

        
        # Normalize pixel intensities
        scaler = sklearn.preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        
        # sparse learning
        coder = SparseCoder(dictionary=D)
        x_train_new=coder.transform(x_train)
        x_test_new=coder.transform(x_test)    
        
        
        # SVM
        model = OneVsRestClassifier(LinearSVC(random_state=0))
        parameters = {'estimator__C':[0.01,0.1,1,10]}
        clf = grid_search.GridSearchCV(model, parameters, score_func=accuracy_score)        
        
        clf.fit(x_train_new, y_train)
        prediction = clf.predict(x_test_new)
        acc = np.sum(prediction == y_test) / float(np.size(y_test))
        print('Test acc rate: %.4f' % acc)

        cv_scores[run,subs] = acc
        #sub5[run] = acc
        
    classification_accuracy = np.mean(cv_scores[:,subs])
    print(classification_accuracy)
    
    target_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\sparse_learning_minbatch_150.mat'
    sio.savemat(target_folder, {'cv_scores':cv_scores})
    
#dict = sio.loadmat(target_folder)
#cv_scores = dict['cv_scores']

np.mean(cv_scores,0)
np.mean(sub5)
