# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:22:26 2024

@author: Amir_chch
"""

import numpy as np
import glob
import os
import scipy.io
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import regex as re
import ast
from statsmodels.stats.anova import AnovaRM
from scipy.stats import wilcoxon,ttest_rel,ttest_ind
import pingouin as pg
import shutil
import seaborn as sn
from statsmodels.stats.multitest import multipletests
import seaborn as sn
from statsmodels.stats.multitest import fdrcorrection
import mne
from scipy.ndimage import shift
#%%
########################################
######## TRF in MNE #############
########################################
from mne.decoding import ReceptiveField
from sklearn.model_selection import KFold
root_path = r'E:\Bonnie\Bonnie\CLOUDS Trinity\dataCND\CombCond\Group*'

conditions = sorted(glob.glob(root_path))

eegfiles  = sorted(glob.glob(root_path+'/pre1hz_pre_pre_dataSub*.mat'))

stimfiles  = sorted(glob.glob(root_path+'/dataStim_pre_pre_dataSub*.mat'))


tmin, tmax = -0.200, 0.600
sfreq=128
sr = ReceptiveField(
    tmin,
    tmax,
    sfreq,
    estimator=1e4,
    scoring="corrcoef",
    patterns=True,
)

n_delays = int((tmax - tmin) * sfreq) + 2


Group1_score = []
Group2_score = []

Group1_coefs =[]
Group2_coefs =[]
for c,cond in enumerate(conditions):
    eegfiles  = sorted(glob.glob(cond+'/pre1hz_pre_pre_dataSub*.mat'))
    stimfiles  = sorted(glob.glob(cond+'/dataStim_pre_pre_dataSub*.mat'))
    
    condnum = os.path.basename(cond)
    allSub_score =[]
    cond_coefs=[]
    for s,eegfile in enumerate(eegfiles):
        print(s)
        eeg = scipy.io.loadmat(eegfile,simplify_cells=True)['eeg']['data']
        stim = scipy.io.loadmat(stimfiles[s],simplify_cells=True)['stim']['data']
        t= scipy.io.loadmat(eegfile,simplify_cells=True)['eeg']
        for tr in range(len(eeg)): ##### making stim and eeg same length
            eeg_tr = eeg[tr]
            stim_tr = stim[tr]
            
            eeg_len = len(eeg_tr)
            stim_len = len(stim_tr)
            
            minlen = np.min([eeg_len,stim_len])
            
            eeg_tr = eeg_tr[:minlen,:]
            stim_tr = stim_tr[:minlen,]
            
            eeg[tr] = eeg_tr
            stim[tr] = stim_tr
            
            
        eeg_all=[]
        stim_all=[]
        for tr in range(len(eeg)): ##### normalizing eeg and stim (finding normalisation factor)
            
            eeg_tr = eeg[tr].flatten()
            stim_tr = stim[tr].flatten()
            
            eeg_all = np.concatenate((eeg_all,eeg_tr),axis=0)
            stim_all = np.concatenate((stim_all,stim_tr),axis=0)
            
        eegnormfactor = np.std(eeg_all)
        stimnormfactor = np.std(stim_all)
        del eeg_all,stim_all,eeg_tr,stim_tr
        for tr in range(len(eeg)): ##### normalizing eeg and stim
            
            eeg[tr] = eeg[tr]/eegnormfactor
            stim[tr] = stim[tr]/stimnormfactor
            
        n_splits = len(eeg)
        cv = KFold(n_splits)
        score = np.zeros((1,n_splits))
        for ii, (train, test) in enumerate(cv.split(stim)):
            
            print("split %s / %s" % (ii + 1, n_splits))
            
            eeg_train=np.array([],ndmin=2)
            stim_train=[]
            
            for tr in range(len(train)):
                if tr==0:
                    eeg_train=eeg[tr]
                else:
                    eeg_train = np.concatenate((eeg_train, eeg[tr]),axis=0)
                    
                stim_train = np.concatenate((stim_train, stim[tr]),axis=0)
            
            
            sr.fit(stim_train.reshape(-1,1), eeg_train)
            
            score[0,test] = sr.score(stim[test][0].reshape(-1,1), eeg[test][0])[0]
            
            eeg_all=np.concatenate((eeg_train, eeg[test][0]),axis=0)
        
        allSub_score.append(np.mean(score,axis=1))
        
        stim_all=np.concatenate((stim_train, stim[test][0]),axis=0)
            
        sr.fit(stim_all.reshape(-1,1), eeg_all)
            
        cond_coefs.append(sr.coef_)   ### weights for all subjects
            
        print(fr' Subject {s} Done')
        
    if condnum=='Group1':
        Group1_score.append(allSub_score)   ### for all 3 condition
        Group1_coefs.append(cond_coefs)
        
    elif condnum=='Group2':
        Group2_score.append(allSub_score)   ### for all 3 condition
        Group2_coefs.append(cond_coefs)
            
    print(fr' Condition {condnum} Done') 
#%%
score_all2 = score_all
all_coefs2=all_coefs
###################                                              
###################    PLOTTING CONDITIONS COMBINED  
###################                            
df_r_2 = pd.DataFrame([])
df_r['CombCondGroup1']=score_all[0]
df_r_2['CombCondGroup2']=score_all[0]
df_r_2=df_r_2.astype(dtype=float)          
fig = plt.figure(figsize=(30,15),dpi=300) 
ax=sn.barplot(data=df_r_2,errorbar='se',width=0.4,
           alpha=0.7)
ax=sn.lineplot(data=df_r_2.T,markers=["s"]*21,palette=['black']*21,alpha=0.1,
            dashes=[(1,1)]*21,legend=False,linestyle='--')
#%%
###################                                              
###################    PLOTTING THE THREE DIFFERENT CONDITIONS  
###################                                              
df_r = pd.DataFrame([])

df_r['cond1'] = score_all[0]
df_r['cond2'] = score_all[1]
df_r['cond3'] = score_all[2]
df=df_r.astype(dtype=float)
FS=15
fig = plt.figure(figsize=(30,15),dpi=300) 
ax=sn.barplot(data=df,errorbar='se',width=0.4,
           alpha=0.7)

ax=sn.lineplot(data=df.T,markers=["s"]*41,palette=['black']*41,alpha=0.1,
            dashes=[(1,1)]*41,legend=False,linestyle='--')
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.xlabel(xlabel='Condition',fontsize=15)
plt.ylabel(ylabel='EEG Pred Corr',fontsize=15)
plt.tight_layout()
wilcoxon(df['cond1'],df_r['cond3'])
#%%
coefs=Group1_coefs[0]
#%%
w_all_sub = np.array([np.squeeze(coefs[i]) for i in range(len(coefs))])
normfactor = np.std(w_all_sub.reshape(len(coefs),-1),axis=1)
normfactor=np.repeat(normfactor.reshape(-1,1),104,axis=1)
w_all_sub_elec_all=[]
w_all_sub2 =[]
for elec in range(32):
    
    w_all_sub_elec = w_all_sub[:,elec,:]/normfactor
    w_all_sub2.append(w_all_sub_elec)
    w_all_sub_elec_all.append(np.mean(w_all_sub_elec,axis=0))
# fig = plt.figure(figsize=(30,15),dpi=300)
w_all_sub2 = np.array(w_all_sub2).T
w_all_sub3 = np.array(w_all_sub_elec_all).T
plt.plot(t,w_all_sub3)
## global field power
t=sr.delays_/128
plt.fill_between(
         x=t,
         y1= np.var(w_all_sub3,axis=1), 
         where=t>=t[0],
         color= "b",
         alpha= 0.2)
plt.title('combined cond Group1',fontsize=20)
plt.xlabel('Time(ms)',fontsize=20)
plt.ylabel('GFP (a.u.)',fontsize=20)
# plt.xticks(ticks=[0,26,53,78,104],labels=[-200,0,200,400,600],fontsize=FS)
plt.yticks(fontsize=20)
plt.tight_layout()
#%%
############ examining envelope trf for group1 and group2

X=1920/2
Y=1080
 
eoi = [12,30,31]
name={12:'Pz',30:'Fz',31:'Cz'}
FS = (30*(X/Y))/1.77
# df_r['subid']=subid
root_path = r'E:\Bonnie\Bonnie\CLOUDS Trinity\dataCND'
conditions=['Cond1','Cond2']
# conditions=['CombCond']
df_all_group1 = pd.DataFrame([])
df_all_group2 = pd.DataFrame([])

for c,cond in enumerate(conditions):
    results = sorted(glob.glob(root_path+fr'/{cond}/Group*/results/v5_21_mTrfRes*group*'))
    root_savepath = fr'E:\Bonnie\Bonnie\CLOUDS Trinity\dataCND\Results\multivariate\v5\{cond}'
    
    df_group1=pd.DataFrame([])
    df_group2=pd.DataFrame([])
    ####### initializing variables
        
    w_all_sub_norm_group1 = np.zeros((20,104,32))
    w_all_sub_norm_group2 = np.zeros((21,104,32))
    
    mw_all_sub_norm_group1 = np.zeros((20,4,104,32)) #### be carfull nuber of features
    mw_all_sub_norm_group2 = np.zeros((21,4,104,32))
    
    mw_all_sub_norm_group1 = np.zeros((20,4,104,32)) #### be carfull nuber of features
    mw_all_sub_norm_group2 = np.zeros((21,4,104,32))
    
    r_all_sub_group1=np.zeros((20,32))
    r_all_sub_group2=np.zeros((21,32))
    
    df_r_group1 = pd.DataFrame([])
    df_w_group1 = pd.DataFrame([])
    df_se_group1 = pd.DataFrame([])
    
    df_r_group2 = pd.DataFrame([])
    df_w_group2 = pd.DataFrame([])
    df_se_group2 = pd.DataFrame([])

    

    



    for elec in range(32):
        # elec=31## cz electrode ## : all
        
        df_stim_group1=pd.DataFrame([])
        df_stim_group2=pd.DataFrame([])
        for k,f in enumerate(results):
            
            group = Path(f).parts[6]
            
            stimindex = Path(f).parts[-1][3:5]
            
                
            
            r = scipy.io.loadmat(f,simplify_cells=True)
            
            t = r['avgModel']['t']
            
            w = r['avgModel']['w']
            
            sub = sorted(glob.glob(root_path+fr'/{cond}/{group}/pre_pre*'))
            if group=='Group1':
                subidgroup1 = [Path(sub[i]).parts[7][-7:-4] for i in range(len(sub))]
                df_group1['Sub'] = subidgroup1
                df_group1['Group'] = [group]*len(subidgroup1)
                df_group1['Condition'] = [cond]*len(subidgroup1)
                df_group1['stimindx'] = [stimindex]*len(subidgroup1)
                df_group1['Correlation'] = r['rAll']
                df_stim_group1=pd.concat((df_stim_group1,df_group1))
                tmp = [r['statsAll'][i]['r'] for i in range(len(r['statsAll']))]
                maxtmp = [np.max(np.mean(tmp[i],axis=0),axis=0) for i in range(len(tmp))]
                r_all_sub_group1[:,:] = np.array(maxtmp)
                
            if group=='Group2':
                subidgroup2 = [Path(sub[i]).parts[7][-7:-4] for i in range(len(sub))]
                df_group2['Sub'] = subidgroup2
                df_group2['Group'] = [group]*len(subidgroup2)
                df_group2['Condition'] = [cond]*len(subidgroup2)
                df_group2['stimindx'] = [stimindex]*len(subidgroup2)
                df_group2['Correlation'] = r['rAll']
                df_stim_group2=pd.concat((df_stim_group2,df_group2))
                tmp = [r['statsAll'][i]['r'] for i in range(len(r['statsAll']))]
                maxtmp = [np.max(np.mean(tmp[i],axis=0),axis=0) for i in range(len(tmp))]
                r_all_sub_group2[:,:] = np.array(maxtmp)
            
            

    # df_all_group1=pd.concat((df_all_group1,df_group1))
    # df_all_group2=pd.concat((df_all_group2,df_group2))
            
            
            w_all_sub = np.array([r['modelAll'][i]['w'] for i in range(len(r['modelAll']))])
            
            if len(w_all_sub.shape)>3: ### multivariate results
                mtrf=1
                normfactor = np.std(w_all_sub.reshape(len(r['modelAll']),-1),axis=1)
                normfactor=np.repeat(normfactor.reshape(-1,1,1),104,axis=2)
                normfactor = np.repeat(normfactor,w_all_sub.shape[1],axis=1)
                w_all_sub_ch = w_all_sub[:,:,:,elec]
                w_all_sub_ch_norm = w_all_sub_ch/normfactor
                
            if len(w_all_sub.shape)==3: ### multivariate results
                mtrf=0
                normfactor = np.std(w_all_sub.reshape(len(r['modelAll']),-1),axis=1)
                normfactor=np.repeat(normfactor.reshape(-1,1),104,axis=1)
                w_all_sub_ch = w_all_sub[:,:,elec]
                w_all_sub_ch_norm = w_all_sub_ch/normfactor
            
            if mtrf==0:
            
                if group=='Group1':
                    w_all_sub_norm_group1[:,:,elec]=w_all_sub_ch_norm
                    
                elif group=='Group2':
                    w_all_sub_norm_group2[:,:,elec] = w_all_sub_ch_norm
                    
            if mtrf==1:
                if group=='Group1':
                    mw_all_sub_norm_group1[:,:,:,elec]=w_all_sub_ch_norm
                    
                elif group=='Group2':
                    mw_all_sub_norm_group2[:,:,:,elec] = w_all_sub_ch_norm
                
            
            if elec not in eoi:
                print('Not necessary for plotting')
                continue
            
            se = np.std(w_all_sub_ch_norm,axis=0)/np.sqrt(len(w_all_sub))
            ###### stats to see if weights are different than zero
            sigt_index=[]
            sigt_time=[]
            p_all=[]
            if mtrf==0:
                for i in range(w_all_sub_ch_norm.shape[1]):
                    
                    stats,pvalue=wilcoxon(w_all_sub_ch_norm[:,i])
                    print('pavlue is : ',pvalue)
                    p_all.append(pvalue)
            if mtrf==1:
                for i in range(w_all_sub_ch_norm.shape[2]):
                    stats,pvalue=wilcoxon(w_all_sub_ch_norm[:,0,i],
                                          w_all_sub_ch_norm[:,1,i]) ##0: env, 1: env'
                    print('pavlue is : ',pvalue)
                    p_all.append(pvalue)
                
            p_allCorr = multipletests(p_all,method='fdr_bh',alpha=0.05)
            sigt=p_allCorr[0]
            
            sigt=sigt*1 #### shift does not work properly on boolean
            
            sigtShift=shift(sigt,1)
            sigtShift=sigtShift.astype(bool)
            sigt = sigt.astype(bool)
            #######################################
            #### finding xmin and xmax for significant time points
            xmin=np.logical_and(np.invert(sigtShift),sigt)
            
            xmax = np.logical_and(sigtShift,np.invert(sigt))
            
            tmin=np.where(xmin!=0)[0]
            tmax=np.where(xmax!=0)[0]
            
            if len(tmin)>len(tmax):
                tmax=np.append(tmax,int(104))
            if mtrf==0:
            ########### PLOTTING AVG TRF WEIGHTS All elecs
                fig = plt.figure(figsize=(30,15),dpi=200) 
            
                ################# fittting image to second monitor
                mngr = plt.get_current_fig_manager()
                posX, posY, sizeX, sizeY = (0,100, 1920, 1080)
                mngr.window.setGeometry(posX, posY, sizeX, sizeY)
                ###################################################
                save_path = root_savepath + fr'/AvgTRFweight_{stimindex}_{group}.png'
                # ax=plt.plot(t,w)
                sn.lineplot(data=w,dashes=False,legend=False,linewidth=3,alpha=.4)
                
                plt.title(fr'{group}',fontsize=FS)
                plt.xlabel('Time(ms)',fontsize=FS)
                plt.ylabel('TRFWeight (a.u.)',fontsize=FS)
                plt.xticks(ticks=[0,26,53,78,104],labels=[-200,0,200,400,600],fontsize=FS)
                plt.yticks(fontsize=FS)
                plt.ylim(-3,3.5)
                plt.tight_layout()
                plt.savefig(save_path,
                                dpi=200,format="png")
                plt.close('all')
            if mtrf==0:
                w_loc = w[:,elec]
                
            if mtrf==1:
                w_loc=pd.DataFrame([])
                w_loc['Env'] = w[0,:,elec] ### 0: Env
                
                w_loc["Env'"] = w[1,:,elec] #### 1: Env'
            if mtrf==0:
                if group=='Group1':
                    
                    df_r_group1[group] = r['rAll']
                    df_w_group1[name[elec]] = w_loc
                    df_se_group1[name[elec]] = se
                    df_group1['Correlation'] = r['rAll']
                    
                    
                elif group=='Group2':
                    
                    df_r_group2[group] = r['rAll']
                    df_w_group2[name[elec]] = w_loc
                    df_se_group2[name[elec]] = se
                    df_group2['Correlation'] = r['rAll']
                    
            # df_w[condition] = w
            # df_w['time'] = t
            ########### PLOTTING TRF WEIGHTS individual elecs
            fig = plt.figure(figsize=(30,15),dpi=200) 
        
            ################# fittting image to second monitor
            mngr = plt.get_current_fig_manager()
            posX, posY, sizeX, sizeY = (0,100, 1920, 1080)
            mngr.window.setGeometry(posX, posY, sizeX, sizeY)
            ###################################################
            save_path = root_savepath + fr'/TRFweight_{stimindex}_{group}_{name[elec]}.png'
            # ax=plt.plot(t,w)
            sn.lineplot(data=w_loc,dashes=False,legend=True,linewidth=3,alpha=.6)
            
            if mtrf==1: ### 0:ENV, 1:ENV'
                plt.fill_between(range(len(t)),w_loc['Env']-se[0],
                                w_loc['Env']+se[0],
                                     color='grey',alpha=0.2,linewidth=3) ### filling betweenu univariate surprisal Model
                plt.fill_between(range(len(t)),w_loc["Env'"]-se[1],
                                 w_loc["Env'"]+se[1],
                                     color='grey',alpha=0.2,linewidth=3) ### filling betweenu univariate surprisal Model
                
            else:
                plt.fill_between(range(len(t)),w_loc-se,
                             w_loc+se,
                                 color='grey',alpha=0.2,linewidth=3) ### filling betweenu univariate surprisal Model
            
            plt.title(fr'{group}',fontsize=FS)
            plt.xlabel('Time(ms)',fontsize=FS)
            plt.ylabel('TRFWeight (a.u.)',fontsize=FS)
            plt.xticks(ticks=[0,26,53,78,104],labels=[-200,0,200,400,600],fontsize=FS)
            plt.yticks(fontsize=FS)
            
            
            for i in range(len(tmin)):
                
                xmin=tmin[i]
                xmax=tmax[i]
                if mtrf==1:
                    y0=min(w_loc.min())-.5
                else:
                    y0=min(w_loc)-.5
                plt.hlines(y=y0,xmin=xmin,xmax=xmax,color='k',linewidth=3.2,alpha=1)
                print(xmin,xmax)
            
            
            plt.ylim(-3,3.5)
            plt.tight_layout()
            plt.savefig(save_path,
                            dpi=200,format="png")
            plt.close('all')
            if mtrf==0:
            ########### PLOTTING GFP WEIGHTS
                fig = plt.figure(figsize=(30,15),dpi=200) 
            
                ################# fittting image to second monitor
                mngr = plt.get_current_fig_manager()
                posX, posY, sizeX, sizeY = (0,100, 1920, 1080)
                mngr.window.setGeometry(posX, posY, sizeX, sizeY)
                ###################################################
                save_path = root_savepath + fr'/GFP_{stimindex}_{group}.png'
                # ax=plt.plot(t,w)
                # sn.lineplot(data=np.var(w,axis=1),dashes=False,legend=False,linewidth=3)
                plt.fill_between(
                        x=t,
                        y1= np.var(w,axis=1), 
                        where=t>=t[0],
                        color= "b",
                        alpha= 0.2)
                plt.title(fr'{group}',fontsize=FS)
                plt.xlabel('Time(ms)',fontsize=FS)
                plt.ylabel('GFP (a.u.)',fontsize=FS)
                # plt.xticks(ticks=[0,26,53,78,104],labels=[-200,0,200,400,600],fontsize=FS)
                plt.yticks(fontsize=FS)
                
                plt.tight_layout()
                plt.savefig(save_path,
                                dpi=200,format="png")
                plt.close('all')
    df_all_group1 = pd.concat((df_all_group1,df_stim_group1))
    df_all_group2 = pd.concat((df_all_group2,df_stim_group2))               
#%%
df_all=pd.concat((df_all_group1,df_all_group2))
df_all=df_all.reset_index(drop=True)
df_soi = df_all.groupby('stimindx')
df_soi=df_soi.get_group('10')
df_soigroup1=df_soi[(df_soi['Group']=='Group1')]
df_soigroup2=df_soi[(df_soi['Group']=='Group2')]
ttest_ind(df_soigroup1['Correlation'],df_soigroup2['Correlation'])
sn.barplot(x='Group', y='Correlation', data=df_all,errorbar='se',width=.5)
a=pg.mixed_anova(data=df_soi,within='Condition',between='Group',
               subject='Sub',dv='Correlation')
#%%
X=1920
Y=1080
FS = (30*(X/Y))/2
fig,ax1 = plt.subplots(figsize=(X,Y),dpi=300)

################# fittting image to second monitor
mngr = plt.get_current_fig_manager()
posX, posY, sizeX, sizeY = (0,100, X, Y)
mngr.window.setGeometry(posX, posY, sizeX, sizeY)
###################################################
ax = sn.barplot(hue='Group',x='Condition', y='Correlation', data=df_all,errorbar='se',width=.5)
plt.legend(loc='upper right')
plt.xticks(fontsize=10)
plt.xlabel('Condition',fontsize=15)
plt.yticks(fontsize=10)
plt.ylabel('Pred corr',fontsize=15)
plt.locator_params(axis='y', nbins=4)
plt.tight_layout()
plt.savefig(r'E:\Bonnie\Bonnie\CLOUDS Trinity\dataCND\Results\multivariate\v5\results\condcombcorr13_2.png',
                dpi=300,format="png")
plt.close()
 #%%
########### MIXED ANOVA ON CORREALTIONS
df_mixedAnova=pd.concat((df_all_group1,df_all_group2))
df_mixedAnova=df_mixedAnova.reset_index(drop=True)
a=pg.mixed_anova(data=df_mixedAnova,within='Condition',between='Group',
               subject='Sub',dv='Correlation')
ax = sn.boxplot(x='condition', y='Correlation', hue='condition', data=df_mixedAnova)
plt.show()
#%%
#################### PLOTTING WEIGHTS ON EACH OTHER ###########################
#%%
##### converting python weights
w_all_sub_norm_group1 =np.squeeze(np.array(Group1_coefs[0])).transpose(0,2,1)
w_all_sub_norm_group2 =np.squeeze(np.array(Group2_coefs[0])).transpose(0,2,1)
root_savepath=r'E:\Bonnie\Bonnie\CLOUDS Trinity\dataCND\Results\pythonResults'
#%%
#####reading channel location and name

ch_loc = scipy.io.loadmat(r'E:\Bonnie\Bonnie\CLOUDS Trinity\chanlocs\chanlocs32_101.mat',
                          simplify_cells=True)

chanlocs=[ch_loc['chanlocs'][i]['labels'] for i in range(len(ch_loc['chanlocs']))]

FS=30


for e in range(w_all_sub_norm_group1.shape[2]):
    
    elecname = chanlocs[e]
    
    w_all_sub_eoi_group1 = w_all_sub_norm_group1[:,:,e]
    w_all_sub_eoi_group2 = w_all_sub_norm_group2[:,:,e]
    
    df_w = pd.DataFrame([])
    p_all=[]
    for i in range(w_all_sub_eoi_group2.shape[1]):
        
        stats,pvalue=ttest_ind(w_all_sub_eoi_group1[:,i],
                               w_all_sub_eoi_group2[:,i])
        # print('pavlue is : ',pvalue)
        p_all.append(pvalue)
        
    p_allCorr = multipletests(p_all,method='fdr_bh',alpha=0.05)
    sigt=p_allCorr[0]
    sigt=sigt*1 #### shift does not work properly on boolean
    
    sigtShift=shift(sigt,1)
    sigtShift=sigtShift.astype(bool)
    sigt = sigt.astype(bool)
    #######################################
    #### finding xmin and xmax for significant time points
    xmin=np.logical_and(np.invert(sigtShift),sigt)
    
    xmax = np.logical_and(sigtShift,np.invert(sigt))
    
    tmin=np.where(xmin!=0)[0]
    tmax=np.where(xmax!=0)[0]
    
    avg_w_eoi_group1 = np.mean(w_all_sub_eoi_group1,axis=0)
    se_w_eoi_group1 = np.std(w_all_sub_eoi_group1,axis=0)/np.sqrt(len(w_all_sub_eoi_group1))

    avg_w_eoi_group2 = np.mean(w_all_sub_eoi_group2,axis=0)
    se_w_eoi_group2 = np.std(w_all_sub_eoi_group2,axis=0)/np.sqrt(len(w_all_sub_eoi_group2))    
    
    df_w['group1']=avg_w_eoi_group1
    df_w['group2']=avg_w_eoi_group2
    
    # if e not in eoi:
    #     print('not plotting')
    #     continue
    
    ########### PLOTTING TRF WEIGHTS individual elecs
    fig = plt.figure(figsize=(30,15),dpi=200) 

    ################# fittting image to second monitor
    mngr = plt.get_current_fig_manager()
    posX, posY, sizeX, sizeY = (0,100, 1920, 1080)
    mngr.window.setGeometry(posX, posY, sizeX, sizeY)
    ###################################################
    
    sn.lineplot(data=df_w,dashes=False,legend=True,linewidth=3,alpha=.4)
    
    plt.fill_between(range(len(t)), df_w['group1']-se_w_eoi_group1,
                     df_w['group1']+se_w_eoi_group1,
                         color='grey',alpha=0.2,linewidth=3) ### filling betweenu group1 se 
    
    plt.fill_between(range(len(t)), df_w['group2']-se_w_eoi_group2,
                     df_w['group2']+se_w_eoi_group2,
                         color='grey',alpha=0.2,linewidth=3) ### filling betweenu group1 se 
    
    plt.title(fr'{elecname}',fontsize=FS)
    plt.xlabel('Time(ms)',fontsize=FS)
    plt.ylabel('TRFWeight (a.u.)',fontsize=FS)
    plt.xticks(ticks=[0,26,53,78,104],labels=[-200,0,200,400,600],fontsize=FS)
    plt.yticks(fontsize=FS)
    
    for i in range(len(tmin)):
        
        xmin=tmin[i]
        xmax=tmax[i]
        print(xmin,xmax)
        plt.hlines(y=min(df_w['group1'])-.5,xmin=xmin,xmax=xmax,color='k',linewidth=3.2,alpha=1)
     
    if (len(tmin)>0) or (e in eoi):
        save_path = root_savepath + fr'/TRFweights_{elecname}.png'
        plt.tight_layout()
        plt.savefig(save_path,
                        dpi=200,format="png")
        plt.close('all')
    else:
        plt.close('all')
#%%
# wilcoxon(df_r['Cond1'],df_r['Cond2'])
FS=15
fig = plt.figure(figsize=(30,15),dpi=300) 
ax=sn.barplot(data=df_r,errorbar='se',width=0.4,
           alpha=0.7)

ax=sn.lineplot(data=df_r.T,markers=["s"]*41,palette=['black']*41,alpha=0.1,
            dashes=[(1,1)]*41,legend=False,linestyle='--')
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.xlabel(xlabel='Condition',fontsize=15)
plt.ylabel(ylabel='EEG Pred Corr',fontsize=15)
plt.tight_layout()
# plt.show()
#%%%
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
tmp_file = get_tmpfile(r"E:\mb_variation\mb_variation\W2VModel\vectors.bin")
Semanticmodel =  KeyedVectors.load_word2vec_format(tmp_file,binary=True,limit=299887)
#%%
import textgrids
import scipy.signal
##extracting envelope my self
cond=['Cond1','Cond2','Cond3','Cond4']
matnorm=32767
edges=np.linspace(0,40,5)
condAll_pho={} ## bin edges
condAll_wo={}
root_save = fr'E:\Bonnie\Bonnie\Autism_Data\Stimuli_Target'
for C,Cname in enumerate(cond):
    
    root_path = fr'E:\Bonnie\Bonnie\Autism_Data\Stimuli_Target\{Cname}'

    
    condname = Path(root_path).parts[-1]
    
    wavfiles=sorted(glob.glob(root_path+fr'/*.wav'))
    txtgridfiles = sorted(glob.glob(root_path+fr'/*.TextGrid'))
    
    ####### Lexical Surprise
    LexicalSurprise = sorted(glob.glob(root_path+fr'/*_surprise.csv'))
    
    
    print(wavfiles,txtgridfiles,LexicalSurprise)
    
    
    cell = np.empty((10,len(wavfiles)),dtype=object)
    
    
    abe  = np.zeros((len(edges)-1),)
    
    
    
    stim_cond_pho={}
    stim_cond_wo={}
    for i,f in enumerate(wavfiles):
        
        trialname = Path(f).parts[-1][:-4]

        #########################################
        ############ Acoustic features###########
        #########################################
        rate,data = scipy.io.wavfile.read(f)
        data=data/matnorm ##### converting to matread values
        duration = len(data)/rate
        print(duration)
        rate_new  = 128
        
        n_samples = int(rate_new*duration)+1
        
        env = scipy.signal.hilbert(data)
        
        env=np.abs(env)
        
        env_resample = scipy.signal.resample(env,num=n_samples) ### resampling envelope
        env_resample[env_resample<0]=0
        ##### env derivitive
        env_der=np.diff(env_resample)
        env_der[env_der<0]=0
        env_der=np.append(env_der, 0)
        ###### binning
        env_ab=env_resample
        env_ab[env_ab==0]=1e-6
        env_ab=-10*np.log10(env_ab)
        bindxs=np.digitize(env_ab,edges)
        abe  = np.zeros((len(edges)-1,len(env_ab)))
        
        for e in range(len(edges)):
            # print(i)
            
            idx=np.where(bindxs==e)[0]
            
            abe[e-1,idx]=env_resample[idx]
            
        cell[0,i] = env_resample.reshape(-1,1)
        cell[1,i] = env_der.reshape(-1,1)
        cell[2,i] = abe.T/np.max(abe.T,0)
            
################################################################           
        

        #########################################
        ############ lexical features ###########
        #########################################
        
        df_tr_wo=pd.DataFrame([])
        df_tr_pho=pd.DataFrame([])
        
        pho = textgrids.TextGrid(txtgridfiles[i])['MAU'] ### all phonems in trial
        wo = textgrids.TextGrid(txtgridfiles[i])['ORT-MAU']
        
        token_pho = [pho[i].text.split() for i in range(len(pho))]
        token_wo = [wo[i].text.split() for i in range(len(wo))]
        
        tstart_pho = [pho[i].xmin for i in range(len(pho))]
        tend_pho = [pho[i].xmax for i in range(len(pho))]
        
        tstart_wo = [wo[i].xmin for i in range(len(wo))]
        tend_wo = [wo[i].xmax for i in range(len(wo))]
        
        df_tr_pho['token']=token_pho
        df_tr_pho['startTime']=tstart_pho
        df_tr_pho['endTime']=tend_pho
        stim_cond_pho[trialname]=df_tr_pho
        
        df_tr_wo['token']=token_wo
        df_tr_wo['startTime']=tstart_wo
        df_tr_wo['endTime']=tend_wo
        stim_cond_wo[trialname]=df_tr_wo
        
    for tri,tr in enumerate(stim_cond_pho.keys()):
        print(i,tr)
        
        stim_pho = stim_cond_pho[tr]
        stim_wo = stim_cond_wo[tr]
        
        nsamples_pho=int(np.round(128*stim_pho['endTime'].iloc[-1]))
        nsamples_wo=int(np.round(128*stim_wo['endTime'].iloc[-1]))
        
        reflen=len(cell[0,tri])
        
        if nsamples_pho!=reflen:
            nsamples_pho=nsamples_pho+(reflen-nsamples_pho)
        
        phonemevector=np.zeros((nsamples_pho,1))
        
        if nsamples_wo!=reflen:
            nsamples_wo=nsamples_wo+(reflen-nsamples_wo)
        
        wordvector=np.zeros((nsamples_wo,1))
        
        dissimilarityVector = np.zeros((nsamples_wo,1))
        surpriseVector = np.zeros((nsamples_wo,1))
        
        
        df_surprise = pd.read_csv(LexicalSurprise[tri])
        surprise_onsets=df_surprise['startTime']
        
        samples = np.round(np.array(rate_new*surprise_onsets)).astype(int)
        surpriseVector[samples,0]=df_surprise['neg_log']
        dissimilarityVector[samples,0]=df_surprise['SemanticDiss']
    
        for i,c in stim_pho.iterrows():
            
            if c['token'][0]=='<p:>':#len(c['word'])==0:c['word'][0]=='<p:>':
                ##### going to next value
                continue
           
            phindex=int(np.round(c['startTime']*128))
            
            phonemevector[phindex,0]=1
            
        for i,c in stim_wo.iterrows(): #### itterating over words in trial
            
            if len(c['token'])==0:#c['word'][0]=='<p:>':
                ##### going to next value
                continue
            
            woindex=int(np.round(c['startTime']*128))
            
            wordvector[woindex,0]=1
                           
        cell[3,tri]=phonemevector ##### cell
        cell[4,tri]=wordvector ##### cell
        cell[5,tri]=dissimilarityVector ##### cell
        cell[6,tri]=surpriseVector ##### cell
        cell[7,tri] = np.concatenate((cell[0,tri], cell[1,tri]),axis=1) 
        cell[8,tri] = np.concatenate((cell[1,tri], cell[2,tri]),axis=1)
        cell[9,tri] = np.concatenate((cell[0,tri],cell[1,tri],
                                      cell[3,tri],cell[4,tri],cell[5,tri],cell[6,tri]),axis=1)
    save_path=os.path.join(root_save,fr'{Cname}','dataStimall_v1.mat')
    print(fr'******************* Saving {Cname} *************')
    trialidx=list(np.arange(1,11))*1
    trialidx=np.array(trialidx,dtype=object).reshape(1,-1)
    
    condIndx=Cname
    condIndx=np.array(condIndx,dtype=object)
    dict_data_stim = {'names':np.array(['Env',"Env'",'AB_Env',
                                        'phonemeonset','wordonset','Dissimilarity','GPTSurprise',
                                        "Env + Env'","Env'+AB_Env",
                                        "Env+Env'+phonemonset+wordonset+Dissimilarity+GPTSurprise"],
                                           dtype=object),
                          'trialIdxs':trialidx,
                          'condIndx':condIndx,
                          'CondNames':'Passive',
                          'fs':128, 
                           'data':cell}
    scipy.io.savemat(save_path,{'stim':dict_data_stim})
#%%
    cell[3,i] = .reshape(-1,len(edges)-1) ### word onset
    cell[4,i] = .reshape(-1,len(edges)-1) #### phonemonset
        
        
    trialidx=list(np.arange(1,11))*1
    trialidx=np.array(trialidx,dtype=object).reshape(1,-1)
    
    condIndx=np.array([[(i+1)]*10 for i in range(1)]).reshape(1,-1)
    condIndx=np.array(condIndx,dtype=object)
    dict_data_stim = {'names':np.array(['my env', "env'_resample",
                                        "env'_downsample","AB_Env_resample",
                                        "AB_Env_resample2"],
                                           dtype=object),
                          'trialIdxs':trialidx,
                          'condIndx':condIndx,
                          'CondNames':'Passive',
                          'fs':128, 
                           'data':cell}
    scipy.io.savemat(save_path,{'stim':dict_data_stim})
#%%
a=scipy.io.loadmat(r'E:\Bonnie\Bonnie\CLOUDS Trinity\Stimuli_Target\matlab1_10._env_der_resamplemat.mat', simplify_cells=True)
env_der_matlab=a['env_der']



#%%
############################ TOPOGRAPHY TRFWeights and corr GAIN >>>>> Bonnie DATA #########################
#########################################################################################################

from mne.viz import plot_topomap
channel_location = r'E:\Bonnie\Bonnie\Autism_Data\dataCND\chanlocs32Biosemi.mat'
channel_location = scipy.io.loadmat(channel_location, simplify_cells=True)
chanlocs = channel_location['chanlocs']
x_loc = [chanlocs[i]['X']/1000 for i in range(len(chanlocs))]
y_loc = [chanlocs[i]['Y']/1000 for i in range(len(chanlocs))]
z_loc = [chanlocs[i]['Z']/1000 for i in range(len(chanlocs))]

coord=np.array([x_loc,y_loc,z_loc]).T
labels = [chanlocs[i]['labels'] for i in range(len(chanlocs))]
locs=dict(zip(labels,coord))
diglocs=mne.channels.make_dig_montage(ch_pos=locs,coord_frame='head')
#%%
biosemi32locs=mne.channels.make_standard_montage(kind='biosemi32')
a=biosemi32locs._get_ch_pos()
xbiosemi = []
ybiosemi = []
for k in a.keys():
    
    xx=a[k][0]
    xbiosemi.append(xx)
    yy=a[k][1]
    ybiosemi.append(yy)
    
biosemilocs = np.array([xbiosemi,ybiosemi]).T  
#%% TIME SERIES PLOTTING
#######creating mne evoked data
### make montage based on location?????
##creating info instance
#%%

info=mne.create_info(ch_names=labels, sfreq=128,ch_types='eeg') 
info=info.set_montage(diglocs)
toi=t[0::5]/1000
#%%
Group_dict = {'Group1':mw_all_sub_norm_group1,
              'Group2':mw_all_sub_norm_group2}

feature_map={0:'Env',
             1:"Env'"}
plt.rcParams.update({'font.size': 20})
for i,g in enumerate(Group_dict):
    print(g)
    
    w = Group_dict[g]
    
    w = np.mean(w,axis=0)
    
    #### looping over weights:
    for f in range(len(w)):
        
        f_name=feature_map[f]
        
        woi=w[f,:,:].T
        
        ev = mne.EvokedArray(data=woi,info=info,tmin=-.2)
        plt.rcParams.update({'font.size': 20})
        ax=ev.plot_topomap(times=toi,average=.04,
                                nrows=5,ncols=5,
                                scalings=1,vlim=(-2,2),show=False)
        ax.suptitle(fr'{f_name} Weights, {g}')    
        ax.set_figheight(15)
        ax.set_figwidth(35)
        ax.set_constrained_layout('constrained')
        ax.savefig((fr'E:\Bonnie\Bonnie\CLOUDS Trinity\dataCND\Results'
                    fr'\TOPO\{cond}\v5\multivariate\{g}\{f_name}.png'),
                   dpi=300)
        print(f_name,g)
plt.rcParams.update(plt.rcParamsDefault)
#%%
info=mne.create_info(ch_names=labels, sfreq=128,ch_types='eeg') 
info=info.set_montage(biosemi32locs)
ev = mne.EvokedArray(data=w.T,info=info,tmin=-.2)
toi=t[0::5]/1000
ev.plot_topomap(toi,average=.04, nrows=5,ncols=5,scalings=1)

#%% CORR PLOTTING
#######
from mne.viz import plot_topomap
for i in range(len(rall1)):
    
    r1 = rall1[i,:]
    r2 = rall2[i,:]
    ################# fittting image to second monitor
    mngr = plt.get_current_fig_manager()
    posX, posY, sizeX, sizeY = (0,100, 1920, 1080)
    mngr.window.setGeometry(posX, posY, sizeX, sizeY)
    ###################################################
    
    figs,axs=plt.subplots(1,2,sharex=False,sharey=False,figsize=(30,15))
    # data = np.mean(r_all_sub_group1,axis=0)
    im,cm=plot_topomap(r1,pos=info,ch_type='eeg',
                  cmap='hot_r',show=False,sphere=.095,vlim=(0,.1),
                  size=50,res=64,axes=axs[0],names=labels)
    figs.colorbar(im,location='bottom',aspect=3,shrink=.3)
    # data = np.mean(r_all_sub_group2,axis=0)
    im,cm=plot_topomap(r2,pos=info,ch_type='eeg',
                  cmap='hot_r',show=False,sphere=.095,vlim=(0,.1),
                  size=50,res=64,axes=axs[1],names=labels)
    axs[0].set_title('Group1')
    axs[1].set_title('Group2') 
    figs.colorbar(im,location='bottom',aspect=3,shrink=.3)
    figs.savefig((fr'E:\Bonnie\Bonnie\Autism_Data\dataCND\Results\multivariate\v5\corr{i}_ridge_withsurprise.png'))
    plt.close('all')
    plt.rcParams.update(plt.rcParamsDefault)
#%%
import mTRFresults as mr
X=1920/2
Y=1080
FS = (30*(X/Y))/1.77

root_path=r'E:\Bonnie\Bonnie\Autism_Data\dataCND'
path_all_g1 = sorted(glob.glob(root_path+r'\Co*\Group1\results\v5_8_mtrf*group1.mat'))
path_all_g2 = sorted(glob.glob(root_path+r'\Co*\Group2\results\v5_8_mtrf*group2.mat'))
plot=input('DO YOU WANT PLOT TRF WEIGHTS (1:Y, 0:NO): ')

sub1 = sorted(glob.glob(root_path+r'/Cond1/Group1/pre_pre*'))
sub2 = sorted(glob.glob(root_path+r'/Cond2/Group2/pre_pre*'))

subidgroup1 = [Path(sub1[i]).parts[7][-7:-4] for i in range(len(sub1))]
subidgroup2=[Path(sub2[i]).parts[7][-7:-4] for i in range(len(sub2))]
stimdict={'1':'Env','2':"Env'",'3':'AB Env','4':'pho',
         '5':'wo','6':"Env+Env'",'7':"Env'+AB Env",'8':'Env+pho+wo'}
elec_num =np.arange(1,32)
foi=2### 
name={12:'Pz',30:'Fz',31:'Cz'}

df_group1=pd.DataFrame([])
df_group2=pd.DataFrame([])

df_group1_all=pd.DataFrame([])

df_group2_all=pd.DataFrame([])

w_all_group1=np.zeros((len(path_all_g1),len(subidgroup1),3,
                       104,32))### time-lags*num_electrodes

w_all_group2=np.zeros((len(path_all_g2),len(subidgroup2),3,
                       104,32)) ### time-lags*num_electrodes
root_savepath=r'E:\Bonnie\Bonnie\Autism_Data\dataCND\Results\multivariate\v5'
for i,p1 in enumerate(path_all_g1):
    
    
    
    df_plot=pd.DataFrame([])
    
    p2 = path_all_g2[i]
    
    results1 = scipy.io.loadmat(p1,simplify_cells=True)
    results2 = scipy.io.loadmat(p2,simplify_cells=True)
    
    group = Path(p1).parts[6]
    stimindex = Path(p1).parts[-1][3:4] ### feature index
    stimindex=stimdict[stimindex]
    cond=Path(p1).parts[5]  ##### condition name
    
    t = results1['avgModel']['t']
    
    
    ############## Correaltion info Group1
    df_group1['Sub'] = subidgroup1
    df_group1['Group'] = ['Group1']*len(subidgroup1)
    df_group1['Condition'] = [cond]*len(subidgroup1)
    df_group1['stimindx'] = [stimindex]*len(subidgroup1)
    # tmp = [results1['statsAll'][i]['r'] for i in range(len(results1['statsAll']))]
    # maxtmp = np.array([np.max(np.mean(tmp[i],axis=0),axis=0) for i in range(len(tmp))])
    # maxCz=maxtmp[:,31]
    # df_group1['Correlation'] = maxCz
    df_group1['Correlation'] = results1['rAll']
    df_group1_all=pd.concat((df_group1_all,df_group1))
    
    ############## Correaltion info Group1
    df_group2['Sub'] = subidgroup2
    df_group2['Group'] = ['Group2']*len(subidgroup2)
    df_group2['Condition'] = [cond]*len(subidgroup2)
    df_group2['stimindx'] = [stimindex]*len(subidgroup2)
    # tmp = [results2['statsAll'][i]['r'] for i in range(len(results2['statsAll']))]
    # maxtmp = np.array([np.max(np.mean(tmp[i],axis=0),axis=0) for i in range(len(tmp))])
    # maxCz=maxtmp[:,31]
    # df_group2['Correlation'] = maxCz
    df_group2['Correlation'] = results2['rAll']
    df_group2_all=pd.concat((df_group2_all,df_group2))
    
    ##### weights group1
    if plot=='1':
        w1=  np.array([results1['modelAll'][i]['w'] \
                                           for i in range(len(results1['modelAll']))])
            
        normfactor = np.std(w1.reshape(len(results1['modelAll']),-1),axis=1).reshape(-1,1)
        w1_norm=w1.reshape(len(results1['modelAll']),-1)/normfactor
        w1_norm = w1_norm.reshape((len(results1['modelAll']),3,104,32))
        
        w_all_group1[i,:,:,:,:] = w1_norm
        se1=np.std(w1_norm,axis=0)/len(w1_norm)
        # se1=np.std(w1_norm,axis=0)/1
        ##### weights group2    
        w2 =  np.array([results2['modelAll'][i]['w'] \
                                           for i in range(len(results2['modelAll']))])
            
        normfactor = np.std(w2.reshape(len(results2['modelAll']),-1),axis=1).reshape(-1,1)
        w2_norm=w2.reshape(len(results2['modelAll']),-1)/normfactor
        w2_norm = w2_norm.reshape((len(results2['modelAll']),3,104,32))    
        
        w_all_group2[i,:,:,:,:] = w2_norm
        se2=np.std(w2_norm,axis=0)/len(w2_norm)
        # se2=np.std(w2_norm,axis=0)/1
        
        
        ############# plotting TRF weights
       
        for e in elec_num:
            save_path=root_savepath+fr'/{cond}/TRF_compared_between_groups_{foi}_{stimindex}_{[e]}.png'
            w1_eoi = w1_norm[:,:,:,e]
            se_eoi1 = se1[:,:,e]
            
            w2_eoi = w2_norm[:,:,:,e]
            se_eoi2 = se2[:,:,e]
            
            df_plot['Group1']=np.mean(w1_eoi,axis=0)[foi,:]
            df_plot['Group2']=np.mean(w2_eoi,axis=0)[foi,:]
            ##### runing stats on weights between groups
            sigt_index=[]
            sigt_time=[]
            p_all=[]
            for i in range(w1_eoi.shape[2]):
                stats,pvalue=ttest_ind(w1_eoi[:,foi,i],
                                      w2_eoi[:,foi,i]) ##0: env, 1: env'
                print('pavlue is : ',pvalue)
                p_all.append(pvalue)
            
            p_allCorr = multipletests(p_all,method='fdr_bh',alpha=0.05)
            sigt=p_allCorr[0]
            
            sigt=sigt*1 #### shift does not work properly on boolean
            
            sigtShift=shift(sigt,1)
            sigtShift=sigtShift.astype(bool)
            sigt = sigt.astype(bool)
            #######################################
            #### finding xmin and xmax for significant time points
            xmin=np.logical_and(np.invert(sigtShift),sigt)
            
            xmax = np.logical_and(sigtShift,np.invert(sigt))
            
            tmin=np.where(xmin!=0)[0]
            tmax=np.where(xmax!=0)[0]
                
            if len(tmin)>len(tmax):
                tmax=np.append(tmax,int(104))
            
            
            ########### PLOTTING TRF WEIGHTS individual elecs
            fig = plt.figure(figsize=(30,15),dpi=200) 
        
            ################# fittting image to second monitor
            mngr = plt.get_current_fig_manager()
            posX, posY, sizeX, sizeY = (0,100, 1920, 1080)
            mngr.window.setGeometry(posX, posY, sizeX, sizeY)
            ###################################################
            
            # ax=plt.plot(t,w)
            sn.lineplot(data=df_plot,dashes=False,legend=True,linewidth=3,alpha=.6)
            
            plt.fill_between(range(len(t)),df_plot['Group1']-se_eoi1[foi,:],
                            df_plot['Group1']+se_eoi1[foi,:],
                                 color='grey',alpha=0.2,linewidth=3) ### filling betweenu univariate surprisal Model
            plt.fill_between(range(len(t)),df_plot['Group2']-se_eoi2[foi,:],
                            df_plot['Group2']+se_eoi2[foi,:],
                                 color='grey',alpha=0.2,linewidth=3)
            
            
            plt.title(fr'{[e]}',fontsize=FS)
            plt.xlabel('Time(ms)',fontsize=FS)
            plt.ylabel('TRFWeight (a.u.)',fontsize=FS)
            plt.xticks(ticks=[0,26,53,78,104],labels=[-200,0,200,400,600],fontsize=FS)
            plt.yticks(fontsize=FS)
            
            
            for i in range(len(tmin)):
                
                xmin=tmin[i]
                xmax=tmax[i]
        
                plt.hlines(y=-.7,xmin=xmin,xmax=xmax,color='k',linewidth=3.2,alpha=1)
                print(xmin,xmax)
            
            
            # plt.ylim(-3,3.5)
            plt.tight_layout()
            plt.savefig(save_path,
                            dpi=200,format="png")
            plt.close()

#%%
################################# reading results using mTRFresults method
###################################################################################
import mTRFresults as mr

channel_location = r'E:\Bonnie\Bonnie\Autism_Data\dataCND\chanlocs32Biosemi.mat'
channel_location = scipy.io.loadmat(channel_location, simplify_cells=True)
chanlocs = channel_location['chanlocs']
labels = [chanlocs[i]['labels'] for i in range(len(chanlocs))]

sub1 = sorted(glob.glob(r'E:\Bonnie\Bonnie\Autism_Data\dataCND\Cond1/Group1/pre_pre*'))
sub2 = sorted(glob.glob(r'E:\Bonnie\Bonnie\Autism_Data\dataCND\Cond1/Group2/pre_pre*'))

subidgroup1 = [Path(sub1[i]).parts[7][-7:-4] for i in range(len(sub1))]
subidgroup2=[Path(sub2[i]).parts[7][-7:-4] for i in range(len(sub2))]

root_path = r'E:\Bonnie\Bonnie\Autism_Data\dataCND\Con*'
resutlts_g1 = sorted(glob.glob(root_path+r'/Group1/results/vall_10_*.mat'))
resutlts_g2 = sorted(glob.glob(root_path+r'/Group2/results/vall_10_*.mat'))

rall1=np.zeros((len(resutlts_g1),32))
rall2=np.zeros((len(resutlts_g2),32))




dfall_corr_g1=pd.DataFrame([])
dfall_corr_g2=pd.DataFrame([])
plot=input('DO YOU WANT PLOT TRF WEIGHTS (1:Y, 0:NO): ')
for i,f1 in enumerate(resutlts_g1):
    
    f2 = resutlts_g2[i]
    
    condname = Path(f1).parts[-4]
    stimindex = Path(f1).parts[-1][5:7]
    
    # featurename = stimdict[stimindex]
    
    save_path = (fr'E:\Bonnie\Bonnie\Autism_Data\dataCND\Results\multivariate\v5'
                 fr'/{condname}')
    
    results1 = mr.mTRFresults(f1)
    featurename = results1.regressor_names
    df_corr_g1 = pd.DataFrame([])
    df_corr_g1['subid'] = subidgroup1
    df_corr_g1['Group'] = ['Group1']*len(subidgroup1)
    df_corr_g1['Condition'] = [condname]*len(subidgroup1)
    df_corr_g1['regressors'] = [featurename]*len(subidgroup1)
    df_corr_g1['Correlation']=results1.get_predcorr(elec='avg')
    dfall_corr_g1=pd.concat((dfall_corr_g1,df_corr_g1))
    # df=pd.DataFrame([])
    # df['Subid']=subidgroup1
    # rallSub1 = pd.DataFrame(results1.get_predcorr(elec='all'))
    
    # df=pd.concat((df,rallSub1),axis=1)
    # df=df.rename(columns=a)
    # df.to_csv(fr'E:\Bonnie\Bonnie\Autism_Data\dataCND\Results\multivariate\v5\Group1_{condname}.csv',index=False)
    
    rall1[i,:] = np.mean(results1.get_predcorr(elec='all'),axis=0)
    
    
    results2 = mr.mTRFresults(f2)
    df_corr_g2 = pd.DataFrame([])
    df_corr_g2['subid'] = subidgroup2
    df_corr_g2['Group'] = ['Group2']*len(subidgroup2)
    df_corr_g2['Condition'] = [condname]*len(subidgroup2)
    df_corr_g2['regressors'] = [featurename]*len(subidgroup2)
    df_corr_g2['Correlation']=results2.get_predcorr(elec='avg')
    dfall_corr_g2=pd.concat((dfall_corr_g2,df_corr_g2))
    
    # df=pd.DataFrame([])
    # df['Subid']=subidgroup2
    # rallSub2= pd.DataFrame(results2.get_predcorr(elec='all'))
    # df=pd.concat((df,rallSub2),axis=1)
    # df=df.rename(columns=a)
    # # df.to_csv(fr'E:\Bonnie\Bonnie\Autism_Data\dataCND\Results\multivariate\v5\Group2_{condname}.csv',index=False)
    rall2[i,:] = np.mean(results2.get_predcorr(elec='all'),axis=0)
#%%
#%%
################################# reading results for comparing shuffle using mTRFresults method
###################################################################################
import mTRFresults as mr

channel_location = r'E:\Bonnie\Bonnie\Autism_Data\dataCND\chanlocs32Biosemi.mat'
channel_location = scipy.io.loadmat(channel_location, simplify_cells=True)
chanlocs = channel_location['chanlocs']
labels = [chanlocs[i]['labels'] for i in range(len(chanlocs))]

sub1 = sorted(glob.glob(r'E:\Bonnie\Bonnie\Autism_Data\dataCND\Cond1/Group1/pre_pre*'))
sub2 = sorted(glob.glob(r'E:\Bonnie\Bonnie\Autism_Data\dataCND\Cond1/Group2/pre_pre*'))

subidgroup1 = [Path(sub1[i]).parts[7][-7:-4] for i in range(len(sub1))]
subidgroup2=[Path(sub2[i]).parts[7][-7:-4] for i in range(len(sub2))]

root_path = r'E:\Bonnie\Bonnie\Autism_Data\dataCND\Con*'
resutlts_g1 = sorted(glob.glob(root_path+r'/Group1/results/vall_10_*.mat'))
results_shuffle_g1 = sorted(glob.glob(root_path+r'/Group1/results/vshuffle_*_*.mat'))


resutlts_g2 = sorted(glob.glob(root_path+r'/Group2/results/vall_10_*.mat'))
results_shuffle_g2 = sorted(glob.glob(root_path+r'/Group2/results/vshuffle_*_*.mat'))

rall1=np.zeros((len(resutlts_g1),32))
rall2=np.zeros((len(resutlts_g2),32))




dfall_corr_g1=pd.DataFrame([])
dfall_corr_g2=pd.DataFrame([])
plot=input('DO YOU WANT PLOT TRF WEIGHTS (1:Y, 0:NO): ')
for i,f1 in enumerate(resutlts_g1):
    
    f1_shuffles = sorted(glob.glob(fr'{os.path.dirname(f1)}/vshuffle_*_*.mat')) 

    f2 = resutlts_g2[i]
    f2_shuffles = sorted(glob.glob(fr'{os.path.dirname(f2)}/vshuffle_*_*.mat'))
    
    condname = Path(f1).parts[-4]
    stimindex = Path(f1).parts[-1][5:7]
    
    # featurename = stimdict[stimindex]
    
    save_path = (fr'E:\Bonnie\Bonnie\Autism_Data\dataCND\Results\multivariate\v5'
                 fr'/{condname}')
    
    results1 = mr.mTRFresults(f1)
    featurename = results1.regressor_names
    df_corr_g1 = pd.DataFrame([])
    df_corr_g1['subid'] = subidgroup1
    df_corr_g1['Group'] = ['Group1']*len(subidgroup1)
    df_corr_g1['Condition'] = [condname]*len(subidgroup1)
    df_corr_g1['regressors'] = [featurename]*len(subidgroup1)
    df_corr_g1['Correlation']=results1.get_predcorr(elec='avg')
    dfall_corr_g1=pd.concat((dfall_corr_g1,df_corr_g1))
    # df=pd.DataFrame([])
    # df['Subid']=subidgroup1
    # rallSub1 = pd.DataFrame(results1.get_predcorr(elec='all'))
    
    # df=pd.concat((df,rallSub1),axis=1)
    # df=df.rename(columns=a)
    # df.to_csv(fr'E:\Bonnie\Bonnie\Autism_Data\dataCND\Results\multivariate\v5\Group1_{condname}.csv',index=False)
    
    rall1[i,:] = np.mean(results1.get_predcorr(elec='all'),axis=0)
    
    
    results2 = mr.mTRFresults(f2)
    df_corr_g2 = pd.DataFrame([])
    df_corr_g2['subid'] = subidgroup2
    df_corr_g2['Group'] = ['Group2']*len(subidgroup2)
    df_corr_g2['Condition'] = [condname]*len(subidgroup2)
    df_corr_g2['regressors'] = [featurename]*len(subidgroup2)
    df_corr_g2['Correlation']=results2.get_predcorr(elec='avg')
    dfall_corr_g2=pd.concat((dfall_corr_g2,df_corr_g2))
    
    # df=pd.DataFrame([])
    # df['Subid']=subidgroup2
    # rallSub2= pd.DataFrame(results2.get_predcorr(elec='all'))
    # df=pd.concat((df,rallSub2),axis=1)
    # df=df.rename(columns=a)
    # # df.to_csv(fr'E:\Bonnie\Bonnie\Autism_Data\dataCND\Results\multivariate\v5\Group2_{condname}.csv',index=False)
    rall2[i,:] = np.mean(results2.get_predcorr(elec='all'),axis=0)

















#%%
#############################
############################# BUTTERFLY PLOTS WEIGHTS
#%%

import mTRFresults as mr
root_path=r'E:\Bonnie\Bonnie\Autism_Data\dataCND'
results1 = sorted(glob.glob(root_path+fr'/Co*/Group1/results/vall_10_mTrfResultsgroup1.mat'))
results2 = sorted(glob.glob(root_path+fr'/Co*/Group2/results/vall_10_mTrfResultsgroup2.mat'))

root_save=r'E:\Bonnie\Bonnie\Autism_Data\dataCND\Results\multivariate\v5'
for i,f1 in enumerate(results1):
    
    f2=results2[i]
    
    condname=Path(f1).parts[-4]
    type_result = Path(f1).parts[-1][0:7] #### be carful for double digit numbers
    # plot1=mr.TRFplot(f1)
    # plot2=mr.TRFplot(f2)
    
    # buttplot1=plot1.plot_weights(show_se=True) #### butterfly plts of weights
    # buttplot2=plot2.plot_weights(show_se=True)
    
    compareplot = mr.TRFplotcompare(f1, f2, elec=[12]) ##### comparing two plots weights for specific electrode
    
    
    # gfp = mr.plot_gfp(f1, f2,avg=True) ####### plotting gfp for both groups
    
    
    
    for k,axe1 in enumerate(compareplot):
        # axe2=buttplot2[k]
        axcompare = compareplot[k]
        # axgfp = axe1
        
        #### butterfly plots parameters
        # save_path1=root_save+fr'/{condname}\{type_result}\Group1'
        # os.makedirs(save_path1,exist_ok=True)
        # save_path2=root_save+fr'/{condname}\{type_result}\Group2'
        # os.makedirs(save_path2,exist_ok=True)
        
        save_pathcompare = root_save+fr'/{condname}\{type_result}\indelectrode'
        os.makedirs(save_pathcompare,exist_ok=True)
        
        # save_pathgfp = root_save+fr'/{condname}\{type_result}\GFP'
        # os.makedirs(save_pathgfp,exist_ok=True)
        
        #### butterfly plots parameters
        # axe1=axe1.set_title(fr'{k}')
        # axe2=axe2.set_title(fr'{k}')
        
        axcompare = axcompare.set_title(fr'{k}_Pz')
        
        # axgfp = axgfp.set_title(fr'{k} GFP')
        
        
        
        #### butterfly plots parameters
        # fig1=axe1.get_figure()
        # fig2=axe2.get_figure()
        
        fig3=axcompare.get_figure()
        # fig4 = axgfp.get_figure()
        
        #### butterfly plots parameters
        # fig1.savefig(save_path1+fr'/{k}_ridge.png',bbox_inches='tight')
        # fig2.savefig(save_path2+fr'/{k}_ridge.png',bbox_inches='tight')
        
        fig3.savefig(save_pathcompare+fr'/{k}.png',bbox_inches='tight')

#%%
r1 = mr.mTRFresults(f1)
a=r1.get_gfp()
b=a[2].reshape(1,-1)
#%%
import mTRFresults as mr
path1 = (r'E:\Bonnie\Bonnie\Autism_Data\dataCND\CombCond\Group1\results'
        r'\v5_1_mTrfResultsgroup1.mat')

path2 = (r'E:\Bonnie\Bonnie\Autism_Data\dataCND\CombCond\Group2\results'
        r'\v5_1_mTrfResultsgroup2.mat')

featname={0:'Env',1:"Env'",2:'Phonemeonset',3:'Wordonset'}
a=mr.TRFplot(path1)
a=a.plot_weights(show_se=True)
# ax=mr.TRFplotcompare(path1, path2, elec=[31])

for i,axe in enumerate(a):
    axe=axe.set_title(fr'{featname[i]}_Cz')
    fig=axe.get_figure()
    
    fig.savefig(fr'E:\Bonnie\Bonnie\Autism_Data\dataCND\Results\multivariate\v5\CombCond\compareuniallg1_cz_{featname[i]}.png',bbox_inches='tight')
    
#%%
def show_figure(fig,num):

    # create a dummy figure and use its
    # manager to display "fig"  
    dummy = plt.figure(num=num,figsize=(30,15),dpi=200)
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
#%%
df_all1=pd.concat((dfall_corr_g1,dfall_corr_g2))
# df_all2=pd.concat((dfall_corr_g1,dfall_corr_g2))
# df_compare = pd.concat((df_all1,df_all2))
df_all1=df_all1.reset_index(drop=True)
df_soi = df_all1.groupby('Condition')
df_soi=df_soi.get_group('Cond3')
df_soigroup1=df_soi[(df_soi['Group']=='Group1')]
df_soigroup2=df_soi[(df_soi['Group']=='Group2')]
ttest_ind(df_soigroup1['Correlation'],df_soigroup2['Correlation'])
sn.barplot(x='regressors',hue='Group', y='Correlation', data=df_all1,width=.5,errorbar='se')
plt.legend(loc='upper right')
plt.xticks(fontsize=15)
plt.xlabel('Condition',fontsize=30)
plt.yticks(fontsize=15)
plt.ylabel('Pred corr',fontsize=30)
plt.locator_params(axis='y', nbins=4)
plt.tight_layout()
plt.xlabel(fontsize=25)
plt.ylabel(fontsize=25)
a=pg.mixed_anova(data=df_all1,within='Condition',between='Group',
               subject='subid',dv='Correlation')
#%%
############### reading text grid files
import textgrids

cond=['Cond1','Cond2','Cond3','Cond4']
cond_all={}
model = 0### 0: phoneme onset model, 1: word onset model
for c in cond:
    root_path=(fr'E:\Bonnie\Bonnie\Autism_Data\Stimuli_Target\{c}')
    condname=Path(root_path).parts[-1]
    files = sorted(glob.glob(root_path+'\*.TextGrid'))
    
    
    stim_cond={}
    for f in files:
        df_tmp=pd.DataFrame([])
        name = Path(f).parts[-1][:-9]
        
        t = textgrids.TextGrid(f)['MAU']
        
        if model==1:
            t = textgrids.TextGrid(f)['ORT-MAU']
            word = [t[i].text.split()[0] for i in range(len(t)) if len(t[i].text.split())!=0]
            
            tstart = [t[i].xmin for i in range(len(t)) if len(t[i].text.split())!=0]
                
            tend = [t[i].xmax for i in range(len(t)) if len(t[i].text.split())!=0]
        
        if model==0:
            t = textgrids.TextGrid(f)['MAU']
            word = [t[i].text.split()[0] for i in range(len(t)) if t[i].text.split()[0]!='<p:>']
            
            tstart = [t[i].xmin for i in range(len(t)) if t[i].text.split()[0]!='<p:>']
                
            tend = [t[i].xmax for i in range(len(t)) if t[i].text.split()[0]!='<p:>']
            
        df_tmp['token']=word
            

            
        
        df_tmp['startTime']=tstart
        df_tmp['endTime']=tend
        df_tmp['speakerTag'] = ['1']*len(tstart)
        # for i,c in df_tmp.iterrows():
        #     if len(c['word'])==0:
        #         df_tmp = df_tmp.drop([i])
                
        #         continue
        stim_cond[name]=df_tmp

    
    cond_all[condname]=stim_cond
#%%
save_path = r'E:\Bonnie\Bonnie\Autism_Data\Stimuli_Target'

for cond in cond_all:
    
    condname=cond
    files = cond_all[condname]
    
    for i,f in enumerate(files):
        print(i,f)
        save_name  = save_path+fr'/{condname}/{f}.csv'
        files[f].to_csv(save_name)
        
        print(fr'saved file {f}')
        
#%% added semantic dissimilarity to surprse csv files
from SemanticDiss import SemDiss
files = sorted(glob.glob('E:\Bonnie\Bonnie\Autism_Data\Stimuli_Target\Con*\*_surprise.csv'))
for f in files:
    condanme = Path(f).parts[-2]
    trialname = Path(f).parts[-1][:-4]
    
    df = SemDiss(f,sr=128)
    
    df.to_csv(fr'{f}')
    print(fr'saving done for {condanme}')
#%%
root_path_datastim=r'E:\Bonnie\Bonnie\Autism_Data\dataCND'
root_path_surprise=r'E:\Bonnie\Bonnie\Autism_Data\Stimuli_Target'
cond=['Cond1','Cond2','Cond3','Cond4']
sr=128
for i,c in enumerate(cond):
    
    
    surppath = sorted(glob.glob(root_path_surprise+fr'/{c}/*_surprise.csv'))
    cell =np.zeros((1,len(surppath)),dtype=object)
    save_name = root_path_surprise+fr'/{c}/datastim_surprise.mat'
    for k,f in enumerate(surppath):
        
        
        
        df = pd.read_csv(f)
        
        onsets=df['startTime']
        vectorlength = int(np.round(sr*df['endTime'].iloc[-1])+1)
       
        surpriseVector = np.zeros((vectorlength,1))
        
        samples = np.round(np.array(sr*onsets)).astype(int)
        surpriseVector[samples,0]=df['neg_log']
        cell[0,k]=surpriseVector
        
    dict_data_stim = {'names':np.array(['GPTSurprise'],
                                               dtype=object),
                              'trialIdxs':'1-10',
                              'condIndx':c,
                              'CondNames':'Passive',
                              'fs':sr, 
                               'data':cell}
    scipy.io.savemat(save_name,{'stim':dict_data_stim})
    
    print(fr'GPT Surprise for condition {c} Created')
    
#%%
from surprisal import surprisal_context_spk
files = sorted(glob.glob('E:\Bonnie\Bonnie\Autism_Data\Stimuli_Target\Con*\*.csv'))

for f in files:
    condanme = Path(f).parts[-2]
    trialname = Path(f).parts[-1][:-4]
    pathway_in = os.path.dirname(f)
    pathway_out = pathway_in

    surprisal_context_spk(files=[fr'{trialname}.csv'],speaker='',context='',
                      context_speaker='',memory=600,name=fr'{trialname}_surprise.csv',pt='',
                      pathway_in=pathway_in,
                      pathway_out=pathway_out)

    print(fr'trial {trialname} from condition {condanme} Done')
#%%
root_savepath=r'E:\Bonnie\Bonnie\Autism_Data\Stimuli_Target'
regname = ['phonemeonset','wordonset']
for cond in cond_all.keys(): ##### looping over conditions (save mat file at end of loop)
    print(cond)
    
    stim = cond_all[cond]
    stimdata=np.zeros((1,len(stim)),dtype=object)
    for tr,data in enumerate(stim): ####looping over trial
        
        onsets = stim[data]
        
        nsamples=int(np.round(128*onsets['endTime'].iloc[-1]))
        
        onsetvector=np.zeros((nsamples,1))
        
        for i,c in onsets.iterrows():
            # # print(c)
            
            # if c['word'][0]=='<p:>':#len(c['word'])==0:c['word'][0]=='<p:>':
            #     ##### going to next value
            #     continue
           
            onsetindex=int(np.round(c['startTime']*128))
            
            onsetvector[onsetindex,0]=1
            
        stimdata[0,tr]=onsetvector
    
    save_path=root_savepath+fr'/{cond}/{regname[model]}_{cond}_testss.mat'
        
    dict_data_stim = {'names':np.array([regname[model]],
                                               dtype=object),
                              'trialIdxs':'1-10',
                              'condIndx':cond,
                              'CondNames':'Passive',
                              'fs':128, 
                               'data':stimdata}
    scipy.io.savemat(save_path,{'stim':dict_data_stim})
    
    print(fr'saving done for {cond}')