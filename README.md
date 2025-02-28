# nervparse
![nervparse(1)](https://github.com/user-attachments/assets/a6c04565-5a11-4511-8323-1564aea91471)


A neural network based dependency parser implemented from the paper **"A Fast and Accurate Dependency Parser using Neural Networks"** by Danqi Chen and Christopher D. Manning.

> [!NOTE]
> **Status: Active Development**
> 
## Project Structure
```
📦 
├─ LICENSE
├─ README.md
├─ data
│  ├─ dev.conll
│  ├─ dev.gold.conll
│  ├─ test.conll
│  ├─ test.gold.conll
│  ├─ train.conll
│  └─ train.gold.conll
├─ emnlp2014-depparser.pdf
├─ requirements.txt
└─ parser
   ├─ __pycache__
   │  ├─ data_utils.cpython-312.pyc
   │  ├─ parser_model.cpython-312.pyc
   │  └─ parser_transitions.cpython-312.pyc
   ├─ parser_eval.py
   ├─ parser_model.pth
   ├─ parser_model.py
   ├─ parser_run.py
   ├─ parser_transitions.py
   └─ parser_utils.py
```

## Run It Yourself
```
git clone https://github.com/vihanvt/nervparse
cd nervparse
```

For training the model according to your own dataset, change the files in data folder and then train and evaluate the model.
```
python parser_run.py
python parser_eval.py
```
## Example Output
![eval](https://github.com/user-attachments/assets/d791f8e3-b27c-4e66-9e64-de8a3ea10526)


## Basic Features
- A multi-layer perceptron architecture for the neural network is used to predict the relations/dependencies between words in sentence to get the structured representation of its grammar.
- The model is trained and tested on English Penn Treebank Dataset, the attached dataset file for training contains 39,832 sentences but due to lack of computing resources, the current version is trained on 1500 sentences only. 
- This is a miniature implementation and it only predicts the dependencies.
- The UAS score for the evaluation is:
  - Dev UAS: 63.63%
  - Test UAS: 64.54%

## Project Roadmap
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
