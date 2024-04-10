# -*- coding: utf-8 -*-
'''
Created on Thu Sep  7 15:00:34 2023

@author: Administrator
'''
import csv
import os
import pandas as pd
import numpy as np
from gpt2probs import gpt2probs
from mistral2probs import mistral2probs



#given a context and a speaker, calculate the surprisal of ONLY the specified speaker using the given file

def surprisal_context_spk_Sara(files, speaker, context, context_speaker, memory, name, pt, pathway_in, pathway_out):
    
    '''This function add some context (perspective taking) 
       to the sentence before computing surprisal values. Based on surprisal_context.py.
       
       - Context: it is sentence used as perspective taking 
       - Memory: it sets a limit to the number of words per sentence
       - Name: it's name for the output file
    '''
    
    # -----------------------------------------------
    # -----------------------------------------------
    # Loop for reading your files

    for filename in files:
        
        list_words = []
        
        # Print some message
        print('.............................')
        print("Starting analysis of ", filename)

        # Read file and create and empty dataframe with column names
        data = pd.read_csv(pathway_in + filename) 
        if context_speaker:
            #data = data.loc[data.speakerTag == speaker] #get the surprisals based on a specified speaker
            df   = pd.DataFrame(columns=['word', 'probability', 'neg_log', 'entropy', 'speakerTag', 'speakerContextTag',
                                         'persp_taken', 'startTime', 'endTime', 'limit_sentence'])
            data = data.reset_index(drop=True)
        else:
            #data = data.loc[data.speakerTag == speaker] #get the surprisals based on a specified speaker
            df   = pd.DataFrame(columns=['word', 'probability', 'neg_log', 'entropy', 'speakerTag', 'startTime', 'endTime', 'limit_sentence'])
            data = data.reset_index(drop=True)
        

        # Loop through rows in each file, discard last row to avoid issues

        length = min(len(data), 5000)
        #length = len(data)
        print(length)
    
        for i, r in data[0:length-1].iterrows():    

            if(i%100==0):
                print(i)
            

            # Select two words: word to start a sentence, and the word to predict
            if(type(data['word'][i+1])!=float):
                word        = data['word'][i+1].lower()
            else:
                word        = data['word'][i+1]
            word_start  = data['word'][i].lower()
            

            speaker = data['speakerTag'][i+1]
            
            onset   = data['startTime'][i+1]
            offset  = data['endTime'][i+1]
            
            #Trying to solve the problem for the startTime and endTime repeated across first and second word

            if i == 0:
                onsetSecondWord = data['startTime'][i+1]
                offsetSecondWord = data['endTime'][i+1]

            # -----------------------------------------------
            # -----------------------------------------------
            # Create a list with your words
            if len(list_words) == 0:

                
            
                # Information for first row
                init_prob = (1/50257)
                init_log  = -np.log(init_prob)
                speaker   = data['speakerTag'][i]
                onset     = data['startTime'][i]
                offset    = data['endTime'][i]
                entropy = 0
                
                # If this is the first word, add the information to the CSV file
                if context_speaker:
                    row = [word_start, init_prob, init_log, entropy, speaker, context_speaker, pt, onset, offset, memory]
                    df.loc[len(df)] = row
                else:
                    row = [word_start, init_prob, init_log, entropy, speaker, onset, offset, memory]
                    df.loc[len(df)] = row
                    

                if context:
                    # Create list with context and first word
                    list_words = [context, word_start]
                else:
                    list_words = [word_start]
            else:
                # Add words to the list
                list_words.append(word_start)

            # -----------------------------------------------
            # -----------------------------------------------
            # Delete the first word of the list 
            if i > (memory - 1):
                list_words.pop(1) # We cannot eliminate the first word after the context

            # Create sentence & obtain probabilities 
            sentence      = (" ".join(list_words))
            #prob, neg_log, entropy = gpt2probs(word, sentence)
            prob, neg_log, entropy = mistral2probs(word, sentence)
        
            if i == 0 :
                onset = onsetSecondWord
                offset = offsetSecondWord


            if context_speaker:
                # Create a new row with our data and append it to our dataframe
                row = [word, prob, neg_log, entropy, speaker, context_speaker, pt, onset, offset, memory]
                df.loc[len(df)] = row
            else:
                # Create a new row with our data and append it to our dataframe
                row = [word, prob, neg_log, entropy, speaker, onset, offset, memory]
                df.loc[len(df)] = row


        # -----------------------------------------------
        # -----------------------------------------------
        # Sort dataframe by index in ascending order, and save it as an CSV file
        df = df.sort_index(ascending=True)  # 
        df.to_csv(pathway_out + 'probs_' + name + filename, index = False)

        print('Analysis of file completed...')
        print('.............................')

        del df, row, filename 



# =============================================================================
# RUN THE FUNCTION
# =============================================================================
# Path to folder where csv files are stored
#input_path = 'C:/Users/Administrator.ADMINTR-703MBN5/OneDrive - Trinity College Dublin/Documents/PhD_Projects/Project1/Code/GPT Surprisal/csvPath_AttentionModel/'
#output_path = 'C:/Users/Administrator.ADMINTR-703MBN5/OneDrive - Trinity College Dublin/Documents/PhD_Projects/Project1/AttentionSwitchExperiment/Probabilities_AttentionModel/'

input_path = 'C:/Users/Administrator.ADMINTR-703MBN5/OneDrive - Trinity College Dublin/Documents/PhD_Projects/Project1/AttentionSwitchExperiment/GPT/Input/transcriptionsReset/'
output_path = 'C:/Users/Administrator.ADMINTR-703MBN5/OneDrive - Trinity College Dublin/Documents/PhD_Projects/Project1/AttentionSwitchExperiment/MISTRAL/Output/Probs_ResetModel/'

# List all files in the folder
fileList = os.listdir(input_path)
# Subject ID (always 1 in this case)
subID = 1
# Path to the folder where the context .csv file is saved (I don't have a context for now, so I leave this empty)
context = ""
# Path and name of the context .csv file (I don't have a context for now, so I leave this empty)
context_speaker = ""
# Limit of words for context (perspective taking)
pt = ""
# Limit of words in sentence (memory)  
memoryLength = 600
# Name for output file
if context:
    name = '_memory' + str(memoryLength) + '_pt' + str(pt) + '-'
else:
    name = '_memory' + str(memoryLength) + '-'
     


surprisal_context_spk_Sara(fileList, subID, context, context_speaker, memoryLength, name, pt, input_path, output_path)
