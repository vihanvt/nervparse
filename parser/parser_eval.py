import os
import random
import torch
from parser_utils import load, vocab
from parser_model import ParserModel
from parser_transitions import PartialParse, minibatch_parse

def evaluate(model, sentences, gold_dependencies, wordvocab, posvocab, labelvocab):
    print("Evaluating model...")
    
    correct_unlabeled = 0
    correct_labeled = 0
    total = 0

    predicted_dependencies = minibatch_parse(sentences, model, batch_size=32, 
                                             wordvocab=wordvocab, posvocab=posvocab, labelvocab=labelvocab)

    for sent_idx, (predicted, gold) in enumerate(zip(predicted_dependencies, gold_dependencies)):
        pred_deps = {}
        for head, dep in predicted:
            pred_deps[dep["index"]] = (head["index"], dep.get("label", "<UNK>"))
        
        for dep_info in gold:
            depid, head_idx = dep_info[0], dep_info[1]
            gold_label = dep_info[2] if len(dep_info) >= 3 else None

            total += 1
            if depid in pred_deps:
                if pred_deps[depid][0] == head_idx:
                    correct_unlabeled += 1
                    if gold_label and pred_deps[depid][1] == gold_label:
                        correct_labeled += 1

    #calculation for uas and las-"not implemented yet"
    uas = (correct_unlabeled / total) * 100 if total > 0 else 0
    las = (correct_labeled / total) * 100 if total > 0 else 0
    
    print(f"Total dependencies: {total}, Correct unlabeled: {correct_unlabeled}, Correct labeled: {correct_labeled}")
    return uas, las

def display(sentence, dependencies):
    
    dep_dict = {}
    for head, dependent in dependencies:
        head_idx = head["index"]
        dep_idx = dependent["index"]
        dep_label = dependent.get("label", "dep")

        if head_idx not in dep_dict:
            dep_dict[head_idx] = []
        dep_dict[head_idx].append((dep_idx, dep_label))

    def build_tree(node, prefix=""):
        """ Recursively prints the tree from the given root node. """
        if node in dep_dict:
            for i, (child, label) in enumerate(dep_dict[node]):
                is_last = (i == len(dep_dict[node]) - 1)
                connector = "└──" if is_last else "├──"
                print(f"{prefix}{connector} {label} → {sentence[child]['word']}")
                build_tree(child, prefix + ("    " if is_last else "│   "))

    root_idx = next((i for i, token in enumerate(sentence) if token["head"] in [0, -1]), None)
    
    if root_idx is not None:
        print(f"{sentence[root_idx]['word']} (ROOT)")
        build_tree(root_idx)
    else:
        print("Root not found, invalid dependencies!")

data_path = os.path.join(os.path.dirname(__file__), "..", "data")
dev_sentences = load(os.path.join(data_path, "dev.conll"))
test_sentences = load(os.path.join(data_path, "test.conll"))
train_sentences = load(os.path.join(data_path, "train.conll"))
wordvocab, posvocab, labelvocab = vocab(train_sentences)

model_path = os.path.join(os.path.dirname(__file__), "parser_model.pth")
model = ParserModel(len(wordvocab), len(posvocab), len(labelvocab), 50, 20, 20)
model.load_state_dict(torch.load(model_path))
model.eval()

gold_dependencies_dev = [
    [(idx, token["head"] - 1, token["label"]) 
     for idx, token in enumerate(sent) if token["head"] != 0] 
    for sent in dev_sentences
]

gold_dependencies_test = [
    [(idx, token["head"] - 1, token["label"]) 
     for idx, token in enumerate(sent) if token["head"] != 0] 
    for sent in test_sentences
]

uas_dev, las_dev = evaluate(model, dev_sentences, gold_dependencies_dev, wordvocab, posvocab, labelvocab)
uas_test, las_test = evaluate(model, test_sentences, gold_dependencies_test, wordvocab, posvocab, labelvocab)

print(f"Dev UAS: {uas_dev:.2f}%, Dev LAS: {las_dev:.2f}%")
print(f"Test UAS: {uas_test:.2f}%, Test LAS: {las_test:.2f}%")
randid = random.randint(0, len(test_sentences) - 1)
randsent = test_sentences[randid]
randdep = minibatch_parse([randsent], model, batch_size=1, wordvocab=wordvocab, posvocab=posvocab, labelvocab=labelvocab)[0]

print("\nDependency Tree for Sentence:", end=" ")
print(" ".join(randsent[i]["word"] for i in range(len(randsent))))
print("-----------------------------------------------------------------------------------")
display(randsent, randdep)
