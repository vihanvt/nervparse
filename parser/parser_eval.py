import os
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

#calculating the uas and las score for the datasets
    uas = (correct_unlabeled / total) * 100 if total > 0 else 0
    las = (correct_labeled / total) * 100 if total > 0 else 0
    
    print(f"Total dependencies: {total}, Correct unlabeled: {correct_unlabeled}, Correct labeled: {correct_labeled}")
    return uas, las

# Load data
data_path = os.path.join(os.path.dirname(__file__), "..", "data")
dev_sentences = load(os.path.join(data_path, "dev.conll"))
test_sentences = load(os.path.join(data_path, "test.conll"))
train_sentences = load(os.path.join(data_path, "train.conll"))
wordvocab, posvocab, labelvocab = vocab(train_sentences)

#loading the trained model for evalution in eval mode later
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