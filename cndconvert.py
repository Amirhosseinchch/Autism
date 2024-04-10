# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:14:19 2024

@author: Amir_chch
"""
import numpy as np
import pandas as pd
import textgrids
import scipy.signal
import glob
import os
import scipy.io
from pathlib import Path

###############################################################
############## Extracting Features from Audio and text grid files and converting to cnd
###############################################################

class cndformat:
    def __init_(self,path,lexical=False):
        
        self.path = path #### path to dir containing wav files and/or textgrid files
        
        wavfiles = sorted(glob.glob(path,r'/*.wav'))
        
        if lexical==True:
        
            txtgridfiles = sorted(glob.glob(path,r'/*.TextGrid'))
            
            if len(wavfiles)!=len(txtgridfiles): ##### check if number of wav files is equal to textgrid files
                
                raise ValueError('Number of wav files does not match number of TextGrid files')
                
            for i,c in enumerate(wavfiles): ##### check if files are named correctly and have same pattern
                
                wavname = os.path.basename(wavfiles[i]).replace('.wav','')
                txtgridname = os.path.basename(wavfiles[i]).replace('.TextGrid','')
                
                if wavname!=txtgridname: #######
                
                    raise ValueError('Names in wav file do not corespond to TextGrid files')
                
    def get_features(self,features=['ennv','envder','phonemonset','wordonset']):
        
        a=1
        
        return a
       
       