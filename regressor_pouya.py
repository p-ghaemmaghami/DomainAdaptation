
import numpy as np
import scipy.io as sio
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import grid_search
from sklearn.metrics import accuracy_score 
from sklearn import linear_model
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#source = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\haxby_fc7_pca10_feat.mat'
#imagenet_targets = range(7)
#imagenet_targets = np.repeat(imagenet_targets,48)
#imagenet_features = feat
#source = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\haxby_cnn_feat.mat'
source = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\imagenet_pca10(acc=91)_cnn_feat.mat'
dict = sio.loadmat(source)
feat = dict['pca_feat']
imagenet_features = np.concatenate((feat[0,0],feat[0,1],feat[0,2],feat[0,3],feat[0,4],feat[0,5],feat[0,6]))
imagenet_targets = np.concatenate((0*np.ones([feat[0,0].shape[0]]),1*np.ones([feat[0,1].shape[0]]),2*np.ones([feat[0,2].shape[0]]),3*np.ones([feat[0,3].shape[0]]),4*np.ones([feat[0,4].shape[0]]),5*np.ones([feat[0,5].shape[0]]),6*np.ones([feat[0,6].shape[0]]),))

#####################################

### Load subjects datasets ########################################################
source_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\'

subjects = (['subj1','subj2','subj3','subj4','subj6'])
cv_scores=np.zeros([12,5])

for subs in range(len(subjects)):
    
    print(subjects[subs])
    #n_samples = dict['n_samples']
    
    for run in range(12):
    
        #folder = 'c:\\Users\\Pouya\\Documents\\haxby\\'+str(subjects[subs])+'\\_normal_augmentedData_new.mat'
        folder = 'c:\\Users\\Pouya\\Documents\\haxby\\'+str(subjects[subs])+'\\_normal_augmentedData_100_VTC_Native.mat'
        dict = sio.loadmat(folder)
        
        features = dict['features']
        targets = dict['targets']
        targets = targets[0,:]
        
        #features = features[0:864,:]
        #targets = targets[0:864]
        
        test_index = range(run*72,(run+1)*72)
        
        x_test = features[test_index,:]
        y_test = targets[test_index]
        
        numberOfAugmentation = 100
        for i in range(numberOfAugmentation):
            for j in range(len(test_index)):
                features [i*864+test_index[j],:] = np.NAN
                targets [i*864+test_index[j]] = np.NAN
                
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
        
        # Normalize pixel intensities
        #scaler = sklearn.preprocessing.StandardScaler()
        #imagenet_features = scaler.fit_transform(imagenet_features)
        
        # sparse
        from sklearn.decomposition import SparseCoder
        pca = PCA(n_components=100)
        pca.fit(x_train)
        exp = pca.explained_variance_ratio_
        print(np.sum(exp))
        x_train=pca.transform(x_train)    
        x_test=pca.transform(x_test)        
        
        
        # pca
        pca = PCA(n_components=100)
        pca.fit(x_train)
        exp = pca.explained_variance_ratio_
        print(np.sum(exp))
        x_train=pca.transform(x_train)    
        x_test=pca.transform(x_test)
        
        
        # Shufflinig
        #ind = range(len(y_train))
        #np.random.shuffle(ind)
        #y_train=y_train[ind]
        #x_train=x_train[ind,:]
        
        
        # Preparing outputs
        cnn_rep = np.zeros([len(y_train),imagenet_features.shape[1]])
        for i in range(len(y_train)):
            selectedTarget = y_train[i]
            selectedTrainset = imagenet_features[imagenet_targets==selectedTarget,:]

            selectedTrainID = np.random.randint(0,np.size(selectedTrainset,0))
            cnn_rep[i,:] = selectedTrainset[selectedTrainID,:]        
        
        #scaler = sklearn.preprocessing.StandardScaler()
        #outputs = scaler.fit_transform(outputs)
        
        #clustering
                
        #from sklearn.kernel_ridge import KernelRidge
        #from sklearn.grid_search import GridSearchCV
        
        numberOfClusters = 10
        numberOfModels = 10
        #model = np.zeros([numberOfModels,],dtype='object')
        model = np.zeros([numberOfClusters,numberOfModels],dtype='object')
        kmeans = KMeans(n_clusters=numberOfClusters, random_state=0).fit(x_train)
        x_new = kmeans.fit_predict(x_train)
        for i in range(numberOfClusters):
            np.sum(x_new==i)
        for cluster in range(numberOfClusters):
            print(cluster)
        for i in range(numberOfModels):
            #train_size = len(y_train)
            #model[i] = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5, param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})
            model[cluster,i] = linear_model.LinearRegression()
            #model[i].fit(x_train, cnn_rep[:,i])
            model[cluster,i].fit(x_train[693*cluster:693*(cluster+1)-1,:], cnn_rep[693*cluster:693*(cluster+1)-1,i])
        
        pred = np.zeros([x_test.shape[0],numberOfModels])
        for i in range(numberOfModels):
            pred[:,i] = model[i].predict(x_test)
            
        cnn_rep_new = np.zeros([cnn_rep.shape[0],numberOfModels])
        for i in range(numberOfModels):
            cnn_rep_new[:,i] = model[i].predict(x_train)
        
        model = OneVsRestClassifier(LinearSVC(random_state=0))
        parameters = {'estimator__C':[0.01,0.1,1,10]}
        clf = grid_search.GridSearchCV(model, parameters, score_func=accuracy_score)        
        
        clf.fit(cnn_rep_new[:,0:numberOfModels], y_train)
        prediction = clf.predict(pred)
        acc = np.sum(prediction == y_test) / float(np.size(y_test))
        print('Test acc rate: %.4f' % acc)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
  
        
        cv_scores[run,subs] = acc
        
    classification_accuracy = np.mean(cv_scores[:,subs])
    print(classification_accuracy)
    
    target_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\Regressor_Augmented_pca_150.mat'
    sio.savemat(target_folder, {'cv_scores':cv_scores})
    
#dict = sio.loadmat(target_folder)
#cv_scores = dict['cv_scores']

np.mean(cv_scores,0)