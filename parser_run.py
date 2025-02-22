import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import load, vocab
from parser_transitions import PartialParse
from parser_model import ParserModel

def oracle(sentence, gold_dependencies):
    pp = PartialParse(sentence, wordvocab, posvocab, labelvocab)
    transitions = []
    while not (len(pp.stack) == 1 and len(pp.buffer) == 0):
        valid_actions = []
        if pp.buffer:  # Only allow Shift if buffer is not empty
            valid_actions.append("S")
        if len(pp.stack) >= 2:
            valid_actions.extend(["LA", "RA"])
        
        if len(pp.stack) >= 2:
            top = pp.stack[-1]
            second = pp.stack[-2]
    
            # Check for left arc
            if (top["index"], second["index"]) in gold_dependencies:
                transitions.append("LA")
                pp.parse_step("LA")
                continue
            # Check for right arc
            elif (second["index"], top["index"]) in gold_dependencies:
                transitions.append("RA")
                pp.parse_step("RA")
                continue
    
        # Default to Shift if no arc is valid and buffer is not empty
        if pp.buffer:
            transitions.append("S")
            pp.parse_step("S")
        else:
            break  # Stop if buffer is empty and no valid actions remain
    
    return transitions

# Define transition mapping
transition_to_idx = {"S": 0, "LA": 1, "RA": 2}

train_sentences = load("data/train.conll")
wordvocab, posvocab, labelvocab = vocab(train_sentences)
model = ParserModel(
    wordvocabsize=len(wordvocab),
    posvocabsize=len(posvocab),
    labelvocabsize=len(labelvocab),
    worddim=50,
    posdim=20,
    labeldim=20,
    hiddenneurons=200,
    dropout=0.3
)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
num_epochs = 10  

for epoch in range(num_epochs):
    total_loss = 0
    for sentence in train_sentences:
        pp = PartialParse(sentence, wordvocab, posvocab, labelvocab)
        gold_dependencies = [(token["head"] - 1, idx) for idx, token in enumerate(sentence) if token["head"] != 0]
        transitions = oracle(sentence, gold_dependencies)
        features = pp.features()
        wordid = torch.tensor(features["wordid"], dtype=torch.long)
        posid = torch.tensor(features["posid"], dtype=torch.long)
        labelid = torch.tensor(features["labelid"], dtype=torch.long)
        
        # Convert transitions to integers
        transition_ids = [transition_to_idx[t] for t in transitions]
        transitions_tensor = torch.tensor(transition_ids, dtype=torch.long)
        
        logits = model(wordid, posid, labelid)
        loss = criterion(logits, transitions_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_sentences)}")

torch.save(model.state_dict(), "parser_model.pth")
print("Training complete. Model saved to parser_model.pth.")