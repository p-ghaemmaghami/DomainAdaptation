# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:15:11 2015

@author: Pouya
"""


import numpy as np
import scipy.io as sio
from nilearn.input_data import NiftiMasker
import pandas

### Load subjects datasets ########################################################
source_folder = 'C:\\Users\\Pouya\\Documents\\haxby\\'
#source_folder = '/data/pouya/projects/haxby/'
subjects = (['subj1','subj2','subj3','subj4','subj5','subj6'])
subjects = (['subj1','subj2','subj3','subj4','subj6'])

for subs in range(len(subjects)):
    
    print(subjects[subs])
    source = source_folder + str(subjects[subs]) +str('\\')
    
    labels = np.recfromcsv(source +str('labels.txt'), delimiter=" ")
    target = labels['labels']
    
    words = set(['face','cat','bottle','house','scissors','shoe','scrambledpix','chair'])
    condition_mask = [ w in words for w in target ]
    condition_mask = np.array(condition_mask, dtype=bool).T
    target = target[condition_mask]
    
    target_int = pandas.get_dummies(target)
    target_int = target_int.values.argmax(1)   
    
    mask_filename = source + str('VTC_adapted_native.nii.gz')
    #mask_filename = source + str('VTC+LO_native.nii.gz')
    nifti_masker = NiftiMasker(mask_img=mask_filename)
    func_filename = source + str('bold.nii.gz')
    #func_filename = source + str('bold_aligned.nii.gz')
    fmri_masked = nifti_masker.fit_transform(func_filename)
    fmri_masked = fmri_masked[condition_mask]
    fmri_masked = fmri_masked[:,np.all(fmri_masked!=0,axis=0)]
        
    numberOfAugmentation = 10
    fwhm = np.linspace(0, 1, numberOfAugmentation)
    #fwhm = np.random.uniform(0, 1, numberOfAugmentation)
    #fwhm = np.random.poisson(lam=0.5, size=numberOfAugmentation)    
    #fwhm = np.abs(np.random.normal(4, 1, numberOfAugmentation))
    #plt.hist(fwhm)
    fwhm[0]=0
    features=np.zeros([np.size(fmri_masked,axis=0)*numberOfAugmentation,np.size(fmri_masked,axis=1)])
    targets=np.zeros([np.size(fmri_masked,axis=0)*numberOfAugmentation,])
    
    # augmented features
    for i in range(numberOfAugmentation):
        print(str(i))
        targets [i*np.size(fmri_masked,axis=0):(i+1)*np.size(fmri_masked,axis=0),] = target_int
        fmri_masked = []
        nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=False , detrend=True, smoothing_fwhm=fwhm[i])
        func_filename = source + str('bold.nii.gz')
        fmri_masked = nifti_masker.fit_transform(func_filename)
        fmri_masked = fmri_masked[condition_mask]
        fmri_masked = fmri_masked[:,np.all(fmri_masked!=0,axis=0)]
        features[i*np.size(fmri_masked,axis=0):(i+1)*np.size(fmri_masked,axis=0),:] = fmri_masked
        
        
    target_folder = source_folder + str(subjects[subs]) + '\\_uniform_augmentedData_10_VTC_Native.mat'
    sio.savemat(target_folder, dict(features=features,targets=targets))
    
    