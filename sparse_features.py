import numpy as np
import scipy.io as sio
from sklearn.decomposition import SparseCoder
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.decomposition import MiniBatchDictionaryLearning

##
source_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\DECAF\\Analysis\\Movie_Genre_adaptation\\feats.mat'
dict = sio.loadmat(source_folder)
features = dict['features']
MovieFeatures = dict['MovieFeatures']

# Source Domain
dict_sparse = DictionaryLearning(alpha=1, n_components=4, max_iter=1000, verbose=3)
dict_sparse.fit(MovieFeatures)
Ds_0 = dict_sparse.components_
coder = SparseCoder(dictionary = Ds_0)
Rs_0 = coder.transform(MovieFeatures)

# Target Domain
dict_feat = [None] * 30
for subs in range(30):
    print(subs)
    
    feat = features[0,subs]

    #dict_sparse = DictionaryLearning(alpha=0.1, n_components=105, max_iter=10, transform_n_nonzero_coefs=105, verbose=3)
    #dict_sparse = SparsePCA(n_components=105, max_iter=3)
    #dict_sparse = MiniBatchDictionaryLearning(alpha=1, n_components=105, batch_size=10, n_iter=100)
    #dict_sparse.fit(feat)
    #Dt_0 = dict_sparse.components_
    #coder = SparseCoder(dictionary = Dt_0, transform_n_nonzero_coefs=105)
    #Rt_0 = coder.transform(feat)
    dict_sparse = SparsePCA(alpha=1, n_components=105, max_iter=20, verbose=3)
    Rt_0 = dict_sparse.fit_transform(feat)
    
    dict_feat[subs] = Rt_0

target_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\DECAF\\Analysis\\Movie_Genre_adaptation\\feats_trans2.mat'
sio.savemat(target_folder, {'dict_feat':dict_feat,'movie_feat':Rs_0})


















## Music Genre classification
source_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\DECAF\\Analysis\\MusicGenreClassification\\feats.mat'
dict = sio.loadmat(source_folder)
MCA_Ft = dict['MCA_Ft']
MEG_Ft = dict['MEG_Ft']
EEG_Ft = dict['EEG_Ft']

# Source Domain
cmp_num = 4
dict_sparse = DictionaryLearning(alpha=1, n_components=cmp_num, max_iter=100, verbose=3)
dict_sparse.fit(MCA_Ft)
Ds_0 = dict_sparse.components_
coder = SparseCoder(dictionary = Ds_0)
Rs_0 = coder.transform(MCA_Ft)

# Target Domain
cmp_num = 4 #3
MEG_Ft_sparse = np.zeros([30,40,cmp_num])
for subs in range(30):
    print(subs)
    
    feat = MEG_Ft[subs,:,:]
    dict_sparse = SparsePCA(alpha=1, n_components=cmp_num, max_iter=5, verbose=3)
    Rt_0 = dict_sparse.fit_transform(feat)
    
    MEG_Ft_sparse[subs,:,:] = Rt_0
    
# Target Domain
cmp_num = 4 #3
EEG_Ft_sparse = np.zeros([32,40,cmp_num])
for subs in range(32):
    print(subs)
    
    feat = EEG_Ft[subs,:,:]
    dict_sparse = SparsePCA(alpha=1, n_components=cmp_num, max_iter=5, verbose=3)
    Rt_0 = dict_sparse.fit_transform(feat)
    
    EEG_Ft_sparse[subs,:,:] = Rt_0

target_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\DECAF\\Analysis\\MusicGenreClassification\\feats_trans2.mat'
sio.savemat(target_folder, {'MEG_Ft_sparse':MEG_Ft_sparse,'EEG_Ft_sparse':EEG_Ft_sparse,'MCA_Ft_sparse':Rs_0})

















## speed/color classification
source_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\VisionReconstruction\\Analysis\\classification\\mlini_ft.mat'
dict = sio.loadmat(source_folder)
MEG_Ft = dict['MEG_Ft']

# sparsify
cmp_num = 50 #3
MEG_Ft_sparse = np.zeros([4,160,cmp_num])
for subs in range(4):
    print(subs)
    
    feat = MEG_Ft[subs,:,:]
    dict_sparse = SparsePCA(alpha=1, n_components=cmp_num, max_iter=10, verbose=3)
    Rt_0 = dict_sparse.fit_transform(feat)
    
    MEG_Ft_sparse[subs,:,:] = Rt_0
    

target_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\VisionReconstruction\\Analysis\\classification\\mlini_ft_sparse.mat'
sio.savemat(target_folder, {'MEG_Ft_sparse':MEG_Ft_sparse})










## speed/color classification using CSP
from sklearn.cross_validation import ShuffleSplit  # noqa
from mne.decoding import CSP
from sklearn.multiclass import OneVsRestClassifier    
from sklearn.svm import LinearSVC                
from sklearn import grid_search                                  
from sklearn.metrics import accuracy_score
import h5py
from sklearn.svm import SVC  # noqa
from sklearn.cross_validation import LeaveOneOut

source_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\VisionReconstruction\\Analysis\\classification\\prni_ft.mat'
dict = h5py.File(source_folder)

MEG_Ft = dict['MEG_Ft']
MEG_Ft = MEG_Ft[:,:,:,:,:]
MEG_Ft = np.transpose(MEG_Ft)
MEG_Ft_reshaped = np.reshape(MEG_Ft, (4, 160, 306, 4*801))

velocity = dict['velocity']
velocity = velocity[:,:]

color = dict['color']
color = color[:,:]



# sparsify
cmp_num = 50 #3
MEG_Ft_sparse = np.zeros([4,160,cmp_num])
for subs in range(4):
    print(subs)
    
    feat = MEG_Ft[subs,:,:]
    dict_sparse = SparsePCA(alpha=1, n_components=cmp_num, max_iter=10, verbose=3)
    Rt_0 = dict_sparse.fit_transform(feat)
    
    MEG_Ft_sparse[subs,:,:] = Rt_0
    



n_components = 400  # pick some components
svc = SVC(C=1, kernel='linear')
csp = CSP(n_components=n_components, reg='ledoit_wolf')

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(160, 10, test_size=0.2, random_state=42)
cv = LeaveOneOut(160)

scores = []
data = MEG_Ft_reshaped[0,:,:,:]
data = np.random.randn(160,306,4,801)
labels=velocity[0,]
labels=color[0,]

from scipy import stats
data = stats.zscore(data)
data = data[~np.isnan(data)]

where_are_NaNs =np.isnan(data)
data[where_are_NaNs] = 0


labels=velocity[0,]
labels=color[0,]
scores = []
data = MEG_Ft[1,:,:]
for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]

    #X_train = csp.fit_transform(data[train_idx,:,:], y_train)
    #X_test = csp.transform(data[test_idx,:,:])
    X_train = data[train_idx,:]
    X_test = data[test_idx,:]

    # fit classifier
    #svc.fit(X_train, y_train)
    #scores.append(svc.score(X_test, y_test))
    
    # multiclass SVM
    model = OneVsRestClassifier(LinearSVC(random_state=0))
    parameters = {'estimator__C':[0.01,0.1,1,10]}
    clf = grid_search.GridSearchCV(model, parameters, score_func=accuracy_score)

    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
    
np.mean(scores)
            
            #prediction = clf.predict(x_test)
            #cv_scores[subs,mask,run] = np.sum(prediction == y_test) / float(np.size(y_test))    
    







from sklearn.lda import LDA  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa
from mne.decoding import CSP


# Assemble a classifier
svc = LDA()
csp = CSP(n_components=64, reg=None, log=True)

labels=np.zeros([160,])
labels[0:79]=1

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()

# Use scikit-learn Pipeline with cross_val_score function
from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import cross_val_score  # noqa
clf = Pipeline([('CSP', csp), ('SVC', svc)])
scores = cross_val_score(clf, MEG_Ft[0,:,1:10], labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))
                                                          
                                                          
                                                          
from sklearn.multiclass import OneVsRestClassifier    
from sklearn.svm import LinearSVC                
from sklearn import grid_search                                  
from sklearn.metrics import accuracy_score 
model = OneVsRestClassifier(LinearSVC(random_state=0))
parameters = {'estimator__C':[0.01,0.1,1,10]}
clf = grid_search.GridSearchCV(model, parameters, score_func=accuracy_score)
clf = Pipeline([('CSP', csp),('SVC', clf)])
scores = cross_val_score(clf, MEG_Ft[0,:,:], labels, cv=cv, n_jobs=1)

clf.fit(MEG_Ft[0,:,:], labels)





from sklearn.svm import SVC  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa
from mne.decoding import CSP  # noqa

n_components = 10  # pick some components
svc = SVC(C=1, kernel='linear')
csp = CSP(n_components=n_components)
csp = CSP(n_components=n_components, reg='ledoit_wolf')

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
scores = []
epochs_data = MEG_Ft[0,:,:]
epochs_data = np.random.randn(160,306,100)



for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data[train_idx,:,:], y_train)
    X_test = csp.transform(epochs_data[test_idx,:])

    # fit classifier
    svc.fit(X_train, y_train)

    scores.append(svc.score(X_test, y_test))
    
    
    
    
source_folder = 'C:\\Users\\Pouya\\Documents\\MATLAB\\VisionReconstruction\\Analysis\\classification\\mlini_ft.mat'
dict = sio.loadmat(source_folder)
MEG_Ft = dict['MEG_Ft']

# sparsify
cmp_num = 10 #3
MEG_Ft_sparse = np.zeros([4,160,cmp_num])
for subs in range(4):
    print(subs)
    
    feat = MEG_Ft[subs,:,:]

    cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
    for train_idx, test_idx in cv:
        y_train, y_test = labels[train_idx], labels[test_idx]
    
        csp = CSP(n_components=n_components, reg='lws')
        
        X_train = csp.fit_transform(feat[train_idx,:], y_train)
        X_test = csp.transform(feat[test_idx,:])

