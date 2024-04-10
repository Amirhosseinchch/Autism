# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:16:53 2024

@author: Amir_chch
"""

import numpy as np
import pandas as pd
import os
import glob
import scipy.io
import scipy
import matplotlib.pyplot as plt
#%%
class mTRFresults:
    def __init__(self,path):
        
        self.path=path
        data = scipy.io.loadmat(self.path,simplify_cells=True)
        shape = len(data['avgModel']['w'].shape)
        # print(data['avgModel']['w'].shape)
        # print(len(data['avgModel']['w']))
        if shape>2:
            trftype='multi'
        else:
            trftype ='uni'
        self.type = trftype
        self.dir = data['avgModel']['Dir']
        self.fs = data['avgModel']['fs']
        self.t = data['avgModel']['t']
        self.n_subs = len(data['modelAll'])
        if trftype=='multi':
            self.n_regressors = len(data['avgModel']['w'])
        else:
            self.n_regressors=1
        self.n_electrodes = data['avgModel']['w'].shape[-1]
        self.info={'t':self.t,'dir':self.dir,'n_regressors': self.n_regressors,
                   'type':self.type,'fs':self.fs,
                   'n_subs':self.n_subs,'n_electrodes':self.n_electrodes}
        
        
        
        del data

    def get_weights(self,avg=True):
        data = scipy.io.loadmat(self.path,simplify_cells=True)
        
        if avg==True:
            weights=data['avgModel']['w']
            
            return weights
        
        if avg==False: ###### return all subjects NORMALIZED WEIGHTS
            
            weights = np.array([data['modelAll'][i]['w'] \
                                               for i in range(self.n_subs)])
                
            normfactor = np.std(weights.reshape(self.n_subs,-1),axis=1).reshape(-1,1)
            weights=weights.reshape(self.n_subs,-1)/normfactor
            weights = weights.reshape((self.n_subs,self.n_regressors,
                                       len(self.t),self.n_electrodes))
            
            return np.squeeze(weights)
        
        del data
        
    
    def get_predcorr(self,elec='avg'):
        #### slect electrode index as int to return best prediction (lambda-wise) for that electrod
        data = scipy.io.loadmat(self.path,simplify_cells=True)
        
        if elec=='avg':
            
            return data['rAll']
            
        if elec=='all':
            
            tmp = [data['statsAll'][i]['r'] for i in range(self.n_subs)]
            maxtmp = np.array([np.max(np.mean(tmp[i],axis=0),axis=0) \
                               for i in range(len(tmp))])
                
            return maxtmp
        
        else:
            
            tmp = [data['statsAll'][i]['r'] for i in range(self.n_subs)]
            maxtmp = np.array([np.max(np.mean(tmp[i],axis=0),axis=0) \
                               for i in range(len(tmp))])
                
            return maxtmp[:,elec]
            
        
        
