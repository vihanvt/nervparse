import os
import torch
from data_utils import load, vocab
from parser_model import ParserModel
from parser_transitions import PartialParse, minibatch_parse

def evaluate(model, sentences, gold_dependencies, wordvocab, posvocab, labelvocab):
    print("Evaluating model...")
    
    correct_unlabeled = 0
    correct_labeled = 0
    total = 0

    predicted_dependencies = minibatch_parse(sentences, model, batch_size=32, 
                                             wordvocab=wordvocab, posvocab=posvocab, labelvocab=labelvocab)

    for predicted, gold in zip(predicted_dependencies, gold_dependencies):
        gold_heads = {(dep[0], idx) for idx, dep in enumerate(gold) if dep[0] is not None}
        pred_heads = {(head["index"], dep["index"]) for head, dep in predicted}

        correct_unlabeled += len(gold_heads & pred_heads)
        total += len(gold_heads)

        gold_labeled = {(dep[0], idx, dep[1]) for idx, dep in enumerate(gold) if dep[0] is not None}
        pred_labeled = {(head["index"], dep["index"], dep["label"]) for head, dep in predicted}

        correct_labeled += len(gold_labeled & pred_labeled)

    uas = (correct_unlabeled / total) * 100 if total > 0 else 0
    las = (correct_labeled / total) * 100 if total > 0 else 0
    return uas, las

# Load data
data_path = os.path.join(os.path.dirname(__file__), "..", "data")
dev_sentences = load(os.path.join(data_path, "dev.conll"))
dev_sentences=dev_sentences[:10]
test_sentences = load(os.path.join(data_path, "test.conll"))
test_sentences=test_sentences[:10]
train_sentences = load(os.path.join(data_path, "train.conll"))
wordvocab, posvocab, labelvocab = vocab(train_sentences)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "parser_model.pth")
model = ParserModel(len(wordvocab), len(posvocab), len(labelvocab), 50, 20, 20)
model.load_state_dict(torch.load(model_path))
model.eval()

# Prepare gold dependencies
gold_dependencies_dev = [[(token["head"] - 1, idx) for idx, token in enumerate(sent) if token["head"] != 0] for sent in dev_sentences]
gold_dependencies_test = [[(token["head"] - 1, idx) for idx, token in enumerate(sent) if token["head"] != 0] for sent in test_sentences]

# Evaluate
uas_dev, las_dev = evaluate(model, dev_sentences, gold_dependencies_dev, wordvocab, posvocab, labelvocab)
uas_test, las_test = evaluate(model, test_sentences, gold_dependencies_test, wordvocab, posvocab, labelvocab)

print(f"Dev UAS: {uas_dev:.2f}%, Dev LAS: {las_dev:.2f}%")
print(f"Test UAS: {uas_test:.2f}%, Test LAS: {las_test:.2f}%")
