# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:56:12 2024

@author: Amir_chch
"""

import pandas as pd
import numpy as np
import os
import nltk
import glob

def get_onsets(path,wordonset=True):
    
    txtgridfiles = sorted(glob.glob(path+r'/*.TextGrid'))
    
    
        