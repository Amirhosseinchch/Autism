# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:31:26 2023

@author: Amir_chch
"""

import numpy as np
import pandas as pd
import scipy
import glob
import os
#%%
root_path = r'E:\Bonnie\dataCND\Cond1\pre_dataSub101.mat'

stim = scipy.io.loadmat(root_path,simplify_cells=True)