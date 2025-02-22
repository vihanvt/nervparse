import sys
import data_utils
class PartialParse():
    def __init__(self, sentence,wordvocab,posvocab,labelvocab):
        self.sentence = [dict(tok, index=i) for i, tok in enumerate(sentence)]
        self.wordvocab=wordvocab
        self.posvocab=posvocab
        self.labelvocab=labelvocab
        self.stack=[{"word":"ROOT","pos":"ROOT","index":-1}]
        self.buffer=self.sentence[:]
        self.dependencies=[]

    def parse_step(self, transition):
        if transition=="S":
            popped=self.buffer.pop(0)
            self.stack.append(popped)
        if transition=="LA":
            dependent=self.stack.pop(-2)
            self.dependencies.append((self.stack[-1],dependent)) #of the form (head,dependent)
        if transition=="RA":
            dependent=self.stack.pop(-1)
            self.dependencies.append((self.stack[-1],dependent))

    def parse(self, transitions):
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies

    def leftchild(self,id): 
        tok=self.stack[id]
        children=[]
        for head,dependent in self.dependencies:
            if(head["index"]==tok["index"]):
                children.append(dependent)    
        children.sort(key=lambda x: self.sentence.index(x))
        return children[0]["label"] if children else "<NULL>"
                
    def rightchild(self,id):
        tok=self.stack[id]
        children=[]
        for head,dependent in self.dependencies:
            if(head["word"]==tok["word"]):
                children.append(dependent)
        children.sort(key=lambda x: self.sentence.index(x))
        return children[-1]["label"] if children else "<NULL>"
    
    def features(self):
        features = {
            "wordid": [],
            "posid": [],
            "labelid": []
        }

        #getting the top 3 elements of stack,buffer,pos for both,right and left labels for stack words
        for i in range(3):
            if i < len(self.stack):
                word = self.stack[-i-1]["word"]
                pos = self.stack[-i-1]["pos"]
                features["wordid"].append(self.wordvocab.get(word, self.wordvocab["<UNK>"]))#add unk if not found
                features["posid"].append(self.posvocab.get(pos, self.posvocab["<UNK>"]))
                left_label = self.leftchild(-i-1)
                right_label = self.rightchild(-i-1)
                features["labelid"].extend([self.labelvocab.get(left_label, self.labelvocab["<UNK>"]),
                                        self.labelvocab.get(right_label, self.labelvocab["<UNK>"])])
            else:
                features["wordid"].append(self.wordvocab["<NULL>"])
                features["posid"].append(self.posvocab["<NULL>"])
                features["labelid"].extend([self.labelvocab["<NULL>"], self.labelvocab["<NULL>"]])

        # Buffer extraction - next 3 words and pos
        for i in range(3):
            if i < len(self.buffer):
                word = self.buffer[i]["word"]
                pos = self.buffer[i]["pos"]
                features["wordid"].append(self.wordvocab.get(word, self.wordvocab["<UNK>"]))
                features["posid"].append(self.posvocab.get(pos, self.posvocab["<UNK>"]))
            else:
                features["wordid"].append(self.wordvocab["<NULL>"])
                features["posid"].append(self.posvocab["<NULL>"])

        return features
    
def minibatch_parse(sentences, model, batch_size):
    dependencies = []
    partial_parses = [PartialParse(sentence.copy()) for sentence in sentences]
    unfinished = partial_parses.copy()
    
    while unfinished:
        batch = unfinished[:batch_size]
        features_batch = [pp.features() for pp in batch]
        transitions = model.predict(features_batch)  # Pass features to model
        for pp, trans in zip(batch, transitions):
            pp.parse_step(trans)
            if len(pp.stack) == 1 and not pp.buffer:
                unfinished.remove(pp)
    return [pp.dependencies for pp in partial_parses]