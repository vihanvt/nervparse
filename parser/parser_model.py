import argparse
import numpy as np
import torch 
import parser_utils as parser_utils
import parser_transitions
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ParserModel(nn.Module):
    def __init__(self, wordvocabsize, posvocabsize, labelvocabsize, worddim, posdim, labeldim, hiddenneurons=200, dropout=0.3):
        super(ParserModel, self).__init__()
        #input layer of model
        self.wordembed = nn.Embedding(wordvocabsize, worddim).to(device)
        self.posembed = nn.Embedding(posvocabsize, posdim).to(device)
        self.labelembed = nn.Embedding(labelvocabsize, labeldim).to(device)
        #output layer
        inpsize = (6 * worddim) + (6 * posdim) + (6 * labeldim)
        self.hidden = nn.Linear(inpsize, hiddenneurons).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.logits = nn.Linear(hiddenneurons, 3).to(device)

    def forward(self, wordid, posid, labelid):
        #flatten to 1d for input to model
        wordid, posid, labelid = wordid.to(device), posid.to(device), labelid.to(device)
        wordemb = self.wordembed(wordid).view(1, -1)  
        posemb = self.posembed(posid).view(1, -1)     
        labelemb = self.labelembed(labelid).view(1, -1)
        joined = torch.cat([wordemb, posemb, labelemb], dim=1)  
        hidden_out = self.hidden(joined)  
        cube = torch.pow(hidden_out, 3)    
        drop = self.dropout(cube)
        logits = self.logits(drop)        
        return logits
    
    def predict(self, features_batch):
        self.eval()
        try:
            if not features_batch:
                return []
            predictions = []
            with torch.no_grad():
                for features in features_batch:
                    wordid = torch.tensor(features["wordid"], dtype=torch.long).to(device)
                    posid = torch.tensor(features["posid"], dtype=torch.long).to(device)
                    labelid = torch.tensor(features["labelid"], dtype=torch.long).to(device)
                    wordemb = self.wordembed(wordid).view(1, -1)
                    posemb = self.posembed(posid).view(1, -1)
                    labelemb = self.labelembed(labelid).view(1, -1)
                    joined = torch.cat([wordemb, posemb, labelemb], dim=1)
                    hidden_out = self.hidden(joined)
                    cube = torch.pow(hidden_out, 3)
                    logits = self.logits(cube)
                    pred = torch.argmax(logits, dim=1).item()
                    predictions.append(pred)
            return predictions
        except Exception as e:
            print(f"Error in predict method: {e}")
            return [0] * len(features_batch)
