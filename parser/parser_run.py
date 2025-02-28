import torch
import torch.nn as nn
import torch.optim as optim
from parser_utils import load, vocab
from parser_transitions import PartialParse
from parser_model import ParserModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            if (top["index"], second["index"]) in golddep:
                transitions.append("LA")
                pp.parse_step("LA")
                continue
            elif (second["index"], top["index"]) in golddep:
                transitions.append("RA")
                pp.parse_step("RA")
                continue

        if pp.buffer:
            transitions.append("S")
            pp.parse_step("S")
        else:
            break

    print(f"Transitions: {transitions}")
    return transitions

sentences = load("../data/train.conll")
wordvocab, posvocab, labelvocab = vocab(sentences)
sentences=sentences[:10000]
print(f"Training on a small dataset of {len(sentences)} sentences")

model = ParserModel(
    wordvocabsize=len(wordvocab),
    posvocabsize=len(posvocab),
    labelvocabsize=len(labelvocab),
    worddim=50, posdim=20, labeldim=20, hiddenneurons=200, dropout=0.3
).to(device)

def weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
model.apply(weights)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lossfunc = nn.CrossEntropyLoss().to(device)

epochs = 10
converter = {"S": 0, "LA": 1, "RA": 2}

for epoch in range(epochs):
    total_loss = 0
    for ids, sentence in enumerate(sentences):
        print(f"Epoch {epoch + 1}, Processing sentence {ids + 1}/{len(sentences)}")
        pp = PartialParse(sentence, wordvocab, posvocab, labelvocab)
        golddep = [(token["head"] - 1, ids) for ids, token in enumerate(sentence) if token["head"] != 0]

        transitions = oracle(sentence, golddep)
        pp = PartialParse(sentence, wordvocab, posvocab, labelvocab)

        for transition in transitions:
            features = pp.features()
            wordid = torch.tensor(features["wordid"], dtype=torch.long).to(device)
            posid = torch.tensor(features["posid"], dtype=torch.long).to(device)
            labelid = torch.tensor(features["labelid"], dtype=torch.long).to(device)
            target = torch.tensor([converter[transition]], dtype=torch.long).to(device)
            logits = model(wordid, posid, labelid)
            loss = lossfunc(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pp.parse_step(transition)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(sentences)}")

torch.save(model.state_dict(), "parser_model.pth")
print("Training complete. Model saved to parser_model.pth.")
