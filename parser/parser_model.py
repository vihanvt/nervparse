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
        #also flatten to 1d
        wordemb = self.wordembed(wordid).view(1, -1)  
        posemb = self.posembed(posid).view(1, -1)     
        labelemb = self.labelembed(labelid).view(1, -1)
        joined = torch.cat([wordemb, posemb, labelemb], dim=1)  
        #adding the cube activation func
        hidden_out = self.hidden(joined)  
        cube = torch.pow(hidden_out, 3)    
        drop = self.dropout(cube)
        logits = self.logits(drop)        
        return logits
    
    def predict(self, features_batch):
        wordid = torch.tensor([feature["wordid"] for feature in features_batch], dtype=torch.long)
        posid = torch.tensor([feature["posid"] for feature in features_batch], dtype=torch.long)
        labelid = torch.tensor([feature["labelid"] for feature in features_batch], dtype=torch.long)

        batch_size = wordid.shape[0]

        wordemb = self.wordembed(wordid).view(batch_size, 6 * 50)
        posemb = self.posembed(posid).view(batch_size, 6 * 20)
        labelemb = self.labelembed(labelid).view(batch_size, 6 * 20)

        joined = torch.cat([wordemb, posemb, labelemb], dim=1)

        with torch.no_grad():
            logits = self.hidden(joined)
            logits = self.logits(logits)
            return torch.argmax(logits, dim=-1).tolist()

