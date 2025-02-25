#this is the file to load conll files and build the dictionary
import os
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch

def load(path):
    sentences = []
    with open(path, 'r', encoding='utf-8') as fob:
        sentence = []
        for line in fob:
            line = line.strip()
            if line:
                parts = line.split("\t")
                if len(parts) == 10:
                    #note to self-check the conll file format for indexing
                    token={'word': parts[1],'pos': parts[3],'head': int(parts[6]),'label': parts[7]}
                    sentence.append(token)
            elif sentence:
                sentences.append(sentence)
                sentence = []
        if sentence:
            sentences.append(sentence)
    return sentences

def vocab(train_data,min_freq=2):
    #words that occur less than twice are ignored
    #the goal is to count all the unique words,labels,pos in the dataset
    #to return - 3 different dicts with seperate counts- dont put in single one
    #self-note: defining 3 counters for words,pos and labels
    wc=Counter()
    pc=Counter()
    lc=Counter()
    for sentence in train_data:
        #adding the respective values using counter.update()
        wc.update([token["word"]for token in sentence])
        pc.update([token["pos"]for token in sentence])
        lc.update([token["label"]for token in sentence])

    wordvocab={"<NULL>":0,"<UNK>":1,"<ROOT>":2}
    posvocab={"<NULL>":0,"<UNK>":1,"<ROOT>":2}
    labelvocab={"<NULL>":0,"<UNK>":1,"<ROOT>":2}

    for i,j in wc.items(): #(i,j)=(words,count)
        if j>=min_freq: 
            wordvocab[i]=len(wordvocab)#to get the each unique word occurence,len is used to give index 
    for i,j in pc.items():
            posvocab[i]=len(posvocab)
    for i,j in lc.items():
            labelvocab[i]=len(labelvocab)
    wordvocabsize=len(wordvocab)
    posvocabsize=len(posvocab)
    labelvocabsize=len(labelvocab)


    return wordvocab,posvocab,labelvocab

     