import argparse
import numpy as np
import torch 
import data_utils
import parser_transitions
import torch.nn as nn
import torch.nn.functional as F

class ParserModel(nn.Module):
    def __init__(self,wordvocabsize,posvocabsize,labelvocabsize,worddim,posdim,labeldim,hiddenneurons=200, dropout=0.3):
        super(ParserModel, self).__init__()
        #input layer of the model
        self.wordembed=nn.Embedding(wordvocabsize,worddim)
        self.posembed=nn.Embedding(posvocabsize,posdim)
        self.labelembed=nn.Embedding(labelvocabsize,labeldim)
        #hidden layer
        inpsize=(6*worddim)+(6*posdim)+(6*labeldim)
        self.hidden=nn.Linear(inpsize,hiddenneurons)
        self.dropout=nn.Dropout(dropout)
        self.logits=nn.Linear(hiddenneurons,3)

    def forward(self,wordid,posid,labelid):
        wordemb=self.wordembed(wordid).view(wordid.size(0), -1)
        posemb=self.posembed(posid).view(wordid.size(0), -1)
        labelemb=self.labelembed(labelid).view(wordid.size(0), -1)
        joined=torch.cat([wordemb,posemb,labelemb],dim=1)

        #activation function=cube activation
        cube=torch.pow(joined,3)
        drop=self.dropout(cube)
        logits=self.logits(drop)
        return logits
    
    def predict(self,features_batch):
        wordid=torch.tensor([feature["wordid"] for feature in features_batch],dtype=torch.long)
        posid=torch.tensor([feature["posid"] for feature in features_batch],dtype=torch.long)
        labelid=torch.tensor([feature["labelid"] for feature in features_batch],dtype=torch.long)
        with torch.no_grad():
            logits=self.forward(wordid,posid,labelid)
            return torch.argmax(logits,dim=-1).tolist()