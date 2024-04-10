# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:40:13 2024

@author: Amir_chch
"""

import numpy as np
import pandas as pd
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from nltk.corpus import stopwords 
#%%

def SemDiss(path,sr,metric='Corr',functionWords=False):
    stop = set(stopwords.words('english'))
    if functionWords==True: ##### if we also want to calculate sementic diss for function words
        stop={}
    tmp_file = get_tmpfile(r"E:\mb_variation\mb_variation\W2VModel\vectors.bin")
    Semanticmodel =  KeyedVectors.load_word2vec_format(tmp_file,binary=True,limit=299887)
    diss_all=[]
    df = pd.read_csv(path)
    
    df['SemanticDiss']=0
    
    length = int(np.round(df.iloc[-1]['endTime']*sr))
    print(length)

    for i,c in df.iterrows():
        # if i==2:
        #     break
        if i==0: ### begining of sentence
            df.at[i,'SemanticDiss']=0
            diss = 0
            diss_all.append(diss)
            
            continue
        
        if c['word'].lower() not in Semanticmodel.index_to_key or \
            c['word'].lower() in stop: ### skipping words that aree not in dict
            
            print('word not in w2v dictionary or is function word')
                
            continue
        
        
        token = c['word']
        token = token.lower()
        
        df_context = df.iloc[:i]
        
        context_words = [w.lower() for w in df_context['word']] ##### all words in previous context
        # print(fr'Context words are :{context_words}')
        contextVectors = [Semanticmodel.get_vector(w).reshape(1,-1) for w in context_words
                          if (w in Semanticmodel.index_to_key) 
                          and (w not in stop)]
        if len(contextVectors)==0:
            print('context words do not exist, skipping')
            df.at[i,'SemanticDiss']=1
            continue
        contextVectors = np.array(contextVectors)
        contextVectorsMean = np.mean(contextVectors,axis=0)
        
        WordVector = Semanticmodel.get_vector(token)
    
        if metric=='Corr':
            # print(token,df_context['word'])
            coef = np.corrcoef(WordVector,contextVectorsMean)[0,1]
            
            diss=1-coef
            # print(diss,i)
            df.at[i,'SemanticDiss']=diss
            diss_all.append(diss)
    return df
        # if metric=='cossdist':
            
        #     coss = np.cos
            