# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:02:32 2024

@author: Amir_chch
"""

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import mTRFresults as mr
import numpy as np
#%%
########################
######################## plotting function
########################

class TRFplot:
    def __init__(self,path):
        
        self.path=path
        
        self.info = mr.mTRFresults(path).info
        
    def plot_weights(self,avg=True,elec='all',show_se=False):
        X=1920/2
        Y=1080
        FS = (30*(X/Y))/1.77
        ########### PLOTTING TRF WEIGHTS individual elecs
        fig = plt.figure(figsize=(30,15),dpi=200) 
        ################# fittting image to second monitor
        mngr = plt.get_current_fig_manager()
        posX, posY, sizeX, sizeY = (0,100, 1920, 1080)
        mngr.window.setGeometry(posX, posY, sizeX, sizeY)
        ###################################################
        
        weights,se = mr.mTRFresults(self.path).get_weights(elec='all')
        weights = np.mean(weights,axis=0)
        if self.info['n_regressors']==1:
            
            weights=np.squeeze(weights)
            se=np.squeeze(se)
            if elec!='all':
                weights=weights[:,:,elec]
                se=se[:,elec]
                weights=np.squeeze(weights)
                se=np.squeeze(se)
            df_plot=pd.DataFrame([])
        
            df_plot['time(ms)']=self.info['t']
            
            df=pd.DataFrame(weights)
            df_plot=pd.concat((df_plot,df),axis=1)
            
            ax1=sn.lineplot(data=df,dashes=False,legend=False,linewidth=3,alpha=.6)
            if show_se==True:
                for e in range( self.info['n_electrodes']):
                    
                    plt.fill_between(range(len(self.info['t'])), df[e]-se[:,e],
                                  df[e]+se[:,e],
                                      color='grey',alpha=0.2,linewidth=3)
            plt.xlabel('Time(ms)',fontsize=FS)
            plt.ylabel('TRFWeight (a.u.)',fontsize=FS)
            plt.xticks(ticks=[0,26,53,78,104],labels=[-200,0,200,400,600],fontsize=FS)
            plt.yticks(fontsize=FS)
            return ax1
        
        if self.info['n_regressors']>1:
            weights=np.squeeze(weights)
            se=np.squeeze(se)
            if elec!='all':
                weights=weights[:,:,elec]
                se=se[:,:,elec]
                # weights=np.squeeze(weights)
                # se=np.squeeze(se)
            df_plot=pd.DataFrame([])
            df_plot['time(ms)']=self.info['t']
            ax_all=[]
            for f in range(self.info['n_regressors']):
                ########### PLOTTING TRF WEIGHTS individual elecs
                fig = plt.figure(figsize=(30,15),dpi=200) 
                ################# fittting image to second monitor
                mngr = plt.get_current_fig_manager()
                posX, posY, sizeX, sizeY = (0,100, 1920, 1080)
                mngr.window.setGeometry(posX, posY, sizeX, sizeY)
                ###################################################
                df=pd.DataFrame(weights[f,:,:]) ### selecting feature
                df_plot=pd.concat((df_plot,df),axis=1)

                print(df.shape)
                print(se.shape)
                ax1=sn.lineplot(data=df,dashes=False,legend=False,linewidth=3,alpha=.6)
                if show_se==True:
                    for e in df.columns:
                        print(e)
                        plt.fill_between(range(len(self.info['t'])), df[e]-se[f,:,e],
                                      df[e]+se[f,:,e],#
                                          color='grey',alpha=0.2,linewidth=3)
                plt.xlabel('Time(ms)',fontsize=FS)
                plt.ylabel('TRFWeight (a.u.)',fontsize=FS)
                plt.xticks(ticks=[0,26,53,78,104],labels=[-200,0,200,400,600],fontsize=FS)
                plt.yticks(fontsize=FS)
                plt.tight_layout()
                print(f)
                ax_all.append(ax1)
                plt.close('all')
                del ax1
            return ax_all
        
            