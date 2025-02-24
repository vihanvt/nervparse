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
        if pp.buffer: 
            valid_actions.append("S")
        if len(pp.stack) >= 2:
            valid_actions.extend(["LA", "RA"])

        if len(pp.stack) >= 2:
            top = pp.stack[-1]
            second = pp.stack[-2]
            #checking the last and second for left arc
            if (top["index"], second["index"]) in gold_dependencies:
                transitions.append("LA")
                pp.parse_step("LA")
                continue
            #check the second and last for right arc
            elif (second["index"], top["index"]) in gold_dependencies:
                transitions.append("RA")
                pp.parse_step("RA")
                continue

        if pp.buffer:
            transitions.append("S")
            pp.parse_step("S")
        else:
            break

    print(f"Transitions: {transitions}")  # Print transitions for debugging
    return transitions

train_sentences = load("../data/train.conll")
wordvocab, posvocab, labelvocab = vocab(train_sentences)
train_sentences = train_sentences[:80]
print(f"Training on a small dataset of {len(train_sentences)} sentences")

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

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
model.apply(init_weights)

#defining loss
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Reduced learning rate
criterion = nn.CrossEntropyLoss()
#training here baby
num_epochs = 10
transition_to_idx = {"S": 0, "LA": 1, "RA": 2}
for epoch in range(num_epochs):
    total_loss = 0
    for idx, sentence in enumerate(train_sentences):
        print(f"Epoch {epoch + 1}, Processing sentence {idx + 1}/{len(train_sentences)}")
        pp = PartialParse(sentence, wordvocab, posvocab, labelvocab)
        gold_dependencies = [(token["head"] - 1, idx) for idx, token in enumerate(sentence) if token["head"] != 0]

        transitions = oracle(sentence, gold_dependencies)
        pp = PartialParse(sentence, wordvocab, posvocab, labelvocab)
        for transition in transitions:
            features = pp.features()
            wordid = torch.tensor(features["wordid"], dtype=torch.long)
            posid = torch.tensor(features["posid"], dtype=torch.long)
            labelid = torch.tensor(features["labelid"], dtype=torch.long)
            target = torch.tensor([transition_to_idx[transition]], dtype=torch.long)
            logits = model(wordid, posid, labelid)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            total_loss += loss.item()
            pp.parse_step(transition)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_sentences)}")
torch.save(model.state_dict(), "parser_model.pth")
print("Training complete. Model saved to parser_model.pth.")