# -*- coding: utf-8 -*-

'''Created on Wed Feb 22 12:44:10 2023
   @author: franklenin.sierra@gmail.com'''

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import math

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model     = AutoModelForCausalLM.from_pretrained("gpt2")



def gpt2probs(word, input_text):
    
    ''' Function to obtain word probabilities from GPT2.
    
    * Inputs: 
        - word: your tartget word (string)
        - input_text: is the sentence where the word is inserted
     
    * Outputs:
        - probability: is the probability of the input word
        - neg_log: is the surprisal of the word, wich is the negative
                   log of the word probability. 
    '''
    
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate output logits
    outputs = model(input_ids)
    logits  = outputs.logits[0, -1, :]  # select last token's logits

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1).detach().numpy()
    #print(probs.shape)

    # Get index of the target word
    idx = tokenizer.encode(word, add_special_tokens=False)

    #Get probability of your word
    probability  = probs[idx[0]]
    neg_log = -np.log(probability)

    #get the entropy of the word 
    entropy = -probability*math.log2(probability)

    logits = outputs.logits[:,-1,:] 
    new_probs = torch.softmax(logits, dim=-1)
    new_entropy = -torch.sum(new_probs*torch.log2(new_probs), dim=-1)

    # Print results
    #print('Target word: ', word)
    #print(f"Probability: {probability}")
    #print(f"Negative log: {neg_log}")

    return(probability, neg_log, new_entropy.item())

