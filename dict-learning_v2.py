import numpy as np
import scipy.io as sio
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import grid_search
from sklearn.metrics import accuracy_score 

from sklearn.decomposition import SparsePCA
from sklearn.decomposition import SparseCoder
from sklearn.decomposition import sparse_encode
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import DictionaryLearning

from nilearn.input_data import NiftiMasker
import pandas

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



####### LOADING SOURCE DATA #################
source = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\imagenet_cnn_feat.mat'
dict = sio.loadmat(source)
feat = dict['feat']
imagenet_features = np.concatenate((feat[0,0],feat[0,1],feat[0,2],feat[0,3],feat[0,4],feat[0,5],feat[0,6]))
imagenet_targets = np.concatenate((0*np.ones([feat[0,0].shape[0]]),1*np.ones([feat[0,1].shape[0]]),2*np.ones([feat[0,2].shape[0]]),3*np.ones([feat[0,3].shape[0]]),4*np.ones([feat[0,4].shape[0]]),5*np.ones([feat[0,5].shape[0]]),6*np.ones([feat[0,6].shape[0]]),))

### Load subjects datasets ########################################################
subjects_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\'

subjects = (['subj1','subj2','subj3','subj4','subj5','subj6'])
masks = (['mask4_vt.nii.gz','VTC_adapted_native.nii.gz'])
mask = 1
cv_scores=np.zeros([12,6])

sparse_components_number = [5,10,20,50,100,200,500,700,1000,1500,2000,3000]



# SPARSITY ON IMAGENET
# SHUFFELING
ind = range(len(imagenet_targets))
np.random.shuffle(ind)
imagenet_targets=imagenet_targets[ind]
imagenet_features=imagenet_features[ind,:]        
    
# Dictionary Learning on Source
sparse_components = 200   
dict_sparse = MiniBatchDictionaryLearning(alpha=1, n_components=sparse_components, verbose=3, batch_size=10, n_iter = 200)
dict_sparse.fit(imagenet_features)
Ds_0 = dict_sparse.components_

coder = SparseCoder(dictionary=Ds_0)
Rs_0 = coder.transform(imagenet_features)

# classification using sparse features
from sklearn import cross_validation
model = OneVsRestClassifier(LinearSVC(random_state=0))
parameters = {'estimator__C':[0.01,0.1,1,10]}
clf = grid_search.GridSearchCV(model, parameters, score_func=accuracy_score)     
scores = cross_validation.cross_val_score(clf, Rs_0, imagenet_targets, cv=10)

#



for subs in range(len(subjects)):
    
    print(subjects[subs])
    
    scaler = sklearn.preprocessing.StandardScaler()
    imagenet_features = scaler.fit_transform(imagenet_features)
        
    source = subjects_folder + str(subjects[subs]) + str('\\')
    
    labels = np.recfromcsv(source +str('labels.txt'), delimiter=" ")
    target = labels['labels']
    
    words = set(['face','cat','bottle','house','scissors','shoe','chair'])
    condition_mask = [ w in words for w in target ]
    condition_mask = np.array(condition_mask, dtype=bool).T
    target = target[condition_mask]
    
    target_int = pandas.get_dummies(target)
    target_int = target_int.values.argmax(1)   
    
    mask_filename = source + str(masks[mask])
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=False , detrend=True)
    func_filename = source + str('bold.nii.gz')
    fmri_masked = nifti_masker.fit_transform(func_filename)
    fmri_masked = fmri_masked[condition_mask]

    fmri_masked = fmri_masked[:,np.all(fmri_masked!=0,axis=0)]
    
    # DEFINING features and targets
    features = fmri_masked
    targets = target_int
    
    # Dictionary Learning on Target
    sparse_components = 200   
    dict_sparse = DictionaryLearning(alpha=1, n_components=sparse_components, max_iter=3, verbose=3)
    dict_sparse.fit(features)
    Dt_0 = dict_sparse.components_
    Rt_0 = sparse_encode(features,dictionary=Dt_0)
    
    # Dictionary Learning on Source iter 2
    sparse_components = 300   
    dict_sparse = MiniBatchDictionaryLearning(alpha=1, n_components=sparse_components, verbose=3, batch_size=10, n_iter = 200)
    dict_sparse.fit(Rs_0)
    Ds_1 = dict_sparse.components_
    #Rs_1 = sparse_encode(Rs_0,dictionary=Ds_1)
    Rt_1 = sparse_encode(Rt_0,dictionary=Ds_1)


    run_range = range(12)
    feat_range = range(756)
    if subs == 4:
        run_range = range(11)
        feat_range = range(693)
 
    for run in run_range:
    
        test_index = range(run*63,(run+1)*63)
        train_index = np.setdiff1d(feat_range,test_index)
        
        x_test = Rt_1[test_index,:]
        y_test = targets[test_index]
                       
        x_train = Rt_1[train_index,:]
        y_train = targets[train_index]
    
        # Normalize pixel intensities
        scaler = sklearn.preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)            
                    
        # SVM
        model = OneVsRestClassifier(LinearSVC(random_state=0))
        parameters = {'estimator__C':[0.01,0.1,1,10]}
        clf = grid_search.GridSearchCV(model, parameters, score_func=accuracy_score)        
        
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)
        acc = np.sum(prediction == y_test) / float(np.size(y_test))
        print('Test acc rate: %.4f' % acc)
        
        cm = confusion_matrix(y_test, prediction)
        cv_scores[run,subs] = acc
        
    classification_accuracy = np.mean(cv_scores[run_range,subs])
    print(classification_accuracy)
 
 

 
 
    
    target_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\dict_learning_v1.mat'
    sio.savemat(target_folder, {'cv_scores':cv_scores})
    
    
    
np.mean(cv_scores[range(12),0])
np.mean(cv_scores[range(12),1])
np.mean(cv_scores[range(12),2])
np.mean(cv_scores[range(12),3])
np.mean(cv_scores[range(11),4])
np.mean(cv_scores[range(12),5]) 





##
target_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\transfer\\imagenet_fc7.mat'
sio.savemat(target_folder, {'imagenet_feat':imagenet_features,'pca_feat':pca_feat,'imagenet_targets':imagenet_targets})


target_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\transfer\\sub1_brain.mat'
sio.savemat(target_folder, {'features':features,'targets':targets})




cm = confusion_matrix(y_test, prediction)
labels=["Bottle", "Cat", "Chair", "Face", "House", "Scissor", "Shoe"]
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted Labels')
plt.ylabel('Ground-Truth LAbels')
for i in range(np.size(cm,0)):
    for j in range(np.size(cm,1)):
        plt.text(j-.2, i+.2, float("{0:.2f}".format(cm[i,j]/float(9))), fontsize=14)
plt.show()