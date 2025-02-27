# nervparse
A neural network based dependency parser implemented from the paper **"A Fast and Accurate Dependency Parser using Neural Networks"** by Danqi Chen and Christopher D. Manning.
> [!NOTE]]
> **Status- Active Development**
> **Not production-ready - created for learning purposes only**
> 
> 
> 
![cover](https://github.com/user-attachments/assets/3fe56a09-31b8-478c-9404-beac07985933)


## Project Structure
```
📦 
├─ LICENSE
├─ README.md
├─ data
│  ├─ dev.conll
│  ├─ dev.gold.conll
│  ├─ test.conll
│  ├─ test.gold.conll
│  ├─ train.conll
│  └─ train.gold.conll
├─ emnlp2014-depparser.pdf
├─ requirements.txt
└─ parser
   ├─ __pycache__
   │  ├─ data_utils.cpython-312.pyc
   │  ├─ parser_model.cpython-312.pyc
   │  └─ parser_transitions.cpython-312.pyc
   ├─ parser_eval.py
   ├─ parser_model.pth
   ├─ parser_model.py
   ├─ parser_run.py
   ├─ parser_transitions.py
   └─ parser_utils.py
```



## Basic Features

◈ A multi-layer perceptron architecture for the neural network is used to predict the relations/dependencies between words in sentence to get the structured representation of its grammar.
◈ The model is trained and tested on English Penn Treebank Dataset, the attached dataset file for training contains 39,832 sentences but due to lack of computing resources, the current version is trained on 1500 sentences only. 
◈ This is a miniature implementation and it only predicts the dependencies.
◈ The UAS score for the evaluation is:
Dev UAS: 63.63%
Test UAS: 64.54%

## Project Flow

- [x] Data Extraction
- [x] Vocabulary Building
- [x] Feature Extraction
- [x] Transition Logic
- [x] Minibatch Parsing
- [x] Model Architecture
- [x] Model Training
- [x] Model Evaluation 
- [ ] Dependency Label Prediction
- [ ] Training on complete dataset
- [ ] Hyperparameter Tuning 
