import sys
import parser_utils as parser_utils
 
class PartialParse():
    def __init__(self, sentence, wordvocab, posvocab, labelvocab):
        self.sentence = [dict(tok, index=i) for i, tok in enumerate(sentence)]
        self.wordvocab = wordvocab
        self.posvocab = posvocab
        self.labelvocab = labelvocab
        self.stack = [{"word":"ROOT", "pos":"ROOT", "label":"ROOT", "index":-1}]
        self.buffer = self.sentence[:]
        self.dependencies = []

    def parse_step(self, transition):
        if transition == "S" or transition == 0:  
            if self.buffer:  
                self.stack.append(self.buffer.pop(0))
        elif transition == "LA" or transition == 1: 
            if len(self.stack) > 1:
                dependent = self.stack.pop(-2)
                head = self.stack[-1]
                depid = dependent["index"]
                if depid >= 0 and depid < len(self.sentence):
                    self.dependencies.append((head, dependent))
                else:
                    dependent["label"] = "LA"
                    self.dependencies.append((head, dependent))
        elif transition == "RA" or transition == 2:
            if len(self.stack) > 1:
                dependent = self.stack.pop()
                head = self.stack[-1]
                depid = dependent["index"]
                if depid >= 0 and depid < len(self.sentence):
                    self.dependencies.append((head, dependent))
                else:
                    dependent["label"] = "RA"
                    self.dependencies.append((head, dependent))

    def parse(self, transitions):
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies

    def leftchild(self, id): 
        tok = self.stack[id]
        children = []
        for head, dependent in self.dependencies:
            if head["index"] == tok["index"]:
                children.append(dependent)    
        if children:
            children.sort(key=lambda x: x["index"])
            return children[0].get("label", "<NULL>")
        return "<NULL>"
                
    def rightchild(self, id):
        tok = self.stack[id]
        children = []
        for head, dependent in self.dependencies:
            if head["index"] == tok["index"]:
                children.append(dependent)
        if children:
            children.sort(key=lambda x: x["index"])
            return children[-1].get("label", "<NULL>")
        return "<NULL>"
    
    def features(self):
        features = {
            "wordid": [],
            "posid": [],
            "labelid": []
        }

        #getting the top 3 elements of stack, buffer, pos for both, right and left labels for stack words
        for i in range(3):
            if i < len(self.stack):
                word = self.stack[-i-1].get("word", "<NULL>")
                pos = self.stack[-i-1].get("pos", "<NULL>")
                features["wordid"].append(self.wordvocab.get(word, self.wordvocab["<UNK>"])) #unique token if nothing found
                features["posid"].append(self.posvocab.get(pos, self.posvocab["<UNK>"]))
                left_label = self.leftchild(-i-1)
                right_label = self.rightchild(-i-1)
                features["labelid"].extend([self.labelvocab.get(left_label, self.labelvocab["<UNK>"]),self.labelvocab.get(right_label, self.labelvocab["<UNK>"])])
            else:
                features["wordid"].append(self.wordvocab["<NULL>"])
                features["posid"].append(self.posvocab["<NULL>"])
                features["labelid"].extend([self.labelvocab["<NULL>"], self.labelvocab["<NULL>"]])

        #buffer extraction - 3 words and pos
        for i in range(3):
            if i < len(self.buffer):
                word = self.buffer[i].get("word", "<NULL>")
                pos = self.buffer[i].get("pos", "<NULL>")
                features["wordid"].append(self.wordvocab.get(word, self.wordvocab["<UNK>"]))
                features["posid"].append(self.posvocab.get(pos, self.posvocab["<UNK>"]))
            else:
                features["wordid"].append(self.wordvocab["<NULL>"])
                features["posid"].append(self.posvocab["<NULL>"])

        return features
    
def minibatch_parse(sentences, model, batch_size, wordvocab, posvocab, labelvocab):
    dependencies = []
    partial_parses = [PartialParse(sentence.copy(), wordvocab, posvocab, labelvocab) for sentence in sentences]
    unfinished = partial_parses.copy()

    maxiter = sum([len(sentence) * 4 for sentence in sentences])  # 4x word count should be enough
    iteration = 0
    try:
        while unfinished and iteration < maxiter:
            iteration += 1
            batch = unfinished[:batch_size]
            if not batch:
                break
            
            features_batch = [pp.features() for pp in batch]
            transitions = model.predict(features_batch)
            #dividing the batches
            new_unfinished = []
            for i, (pp, trans_idx) in enumerate(zip(batch, transitions)):
                trans = ["S", "LA", "RA"][trans_idx] if isinstance(trans_idx, int) else trans_idx
                
                #check for the correct transitions
                valid_transition = False
                if trans == "S" and pp.buffer:
                    valid_transition = True
                elif (trans == "LA" or trans == "RA") and len(pp.stack) >= 2:
                    if trans == "LA" and pp.stack[-2]["index"] != -1:#root cannot be dependent
                        valid_transition = True
                    elif trans == "RA":
                        valid_transition = True
                
                #better idea- do shift and ra as default transitions
                if valid_transition:
                    pp.parse_step(trans)
                elif pp.buffer:
                    pp.parse_step("S")
                elif len(pp.stack) >= 2:
                    pp.parse_step("RA")
                if len(pp.stack) == 1 and not pp.buffer:
                    pass #completed
                else:
                    new_unfinished.append(pp)

            unfinished = new_unfinished + unfinished[len(batch):]

        if iteration >= maxiter or unfinished:
            for pp in unfinished:
                while pp.buffer:
                    pp.parse_step("S")
                while len(pp.stack) > 1:
                    pp.parse_step("RA")
    except Exception as e:
        print(f"Error in minibatch_parse: {e}")
    
    return [pp.dependencies for pp in partial_parses]