import numpy as np
import scipy.io as sio
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import grid_search
from sklearn.metrics import accuracy_score 

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
total_confmat=np.zeros([7,7])

for subs in range(len(subjects)):
    
    print(subjects[subs])
    
    sparse_components = 200
    
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

    # dimensionality scaling
    #pca_feat = imagenet_features
    pca = PCA(n_components=np.size(features,1))
    pca.fit(imagenet_features)
    pca_feat = pca.transform(imagenet_features)
    
    # Shufflinig
    ind = range(len(imagenet_targets))
    np.random.shuffle(ind)
    imagenet_targets=imagenet_targets[ind]
    pca_feat=pca_feat[ind,:]        
    
    # Dictionary Learning on Source   
    dict_sparse = MiniBatchDictionaryLearning(alpha=1, n_components=sparse_components, verbose=3, batch_size=10, n_iter = 1000)
    dict_sparse.fit(pca_feat)
    Ds_0 = dict_sparse.components_
    
    #Rt_1 = sparse_encode(features,dictionary=Ds_0)
    # Dictionary Learning on Target
    dict_sparse = DictionaryLearning(alpha=1, n_components=sparse_components, max_iter=3, verbose=3)
    dict_sparse.fit(features)
    Dt_0 = dict_sparse.components_
    coder = SparseCoder(dictionary=Dt_0)
    Rt_0 = coder.transform(features)
    
    # Target Reconstruction
    Xt_1 = np.mat(Rt_0) * np.mat(Ds_0)
    dict_sparse = DictionaryLearning(alpha=1, n_components=sparse_components, max_iter=3, verbose=3)
    dict_sparse.fit(Xt_1)
    Dt_1 = dict_sparse.components_
    coder = SparseCoder(dictionary=Dt_1)
    Rt_1 = coder.transform(Xt_1)

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
    
        cv_scores[run,subs] = acc

        cm = confusion_matrix(y_test, prediction)
        total_confmat = total_confmat + cm
        
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


np.std(cv_scores[range(12),0])
np.std(cv_scores[range(12),1])
np.std(cv_scores[range(12),2])
np.std(cv_scores[range(12),3])
np.std(cv_scores[range(11),4])
np.std(cv_scores[range(12),5])

np.mean(np.concatenate((cv_scores[range(12),0],cv_scores[range(12),1],cv_scores[range(12),2],cv_scores[range(12),3],cv_scores[range(11),4],cv_scores[range(12),5]),0))

np.std(np.concatenate((cv_scores[range(12),0],cv_scores[range(12),1],cv_scores[range(12),2],cv_scores[range(12),3],cv_scores[range(11),4],cv_scores[range(12),5]),0))


dict = sio.loadmat(target_folder)
cv_scores = dict['cv_scores']

from scipy.stats import ttest_ind
ttest_ind(x, y, equal_var=False)
#Ttest_indResult(statistic=-2.8709266786970122, pvalue=0.0047294403706418572)


##
target_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\transfer\\imagenet_fc7.mat'
sio.savemat(target_folder, {'imagenet_feat':imagenet_features,'pca_feat':pca_feat,'imagenet_targets':imagenet_targets})


target_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\transfer\\sub1_brain.mat'
sio.savemat(target_folder, {'features':features,'targets':targets})




target_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\results\\confmat_sparse.mat'
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
