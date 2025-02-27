# nervparse
A neural network-based dependency parser implemented from the paper **"A Fast and Accurate Dependency Parser using Neural Networks"** by Danqi Chen and Christopher D. Manning.

> [!NOTE]
> **Not production-ready - created for learning purposes only**

## Basic Features
- A multi-layer perceptron architecture for the neural network is used to predict the relations/dependencies between words in a sentence to get the structured representation of its grammar.
- The model is trained and tested on the English Penn Treebank Dataset. The attached dataset file for training contains **39,832 sentences**, but due to lack of computing resources, the current version is trained on **1,500 sentences only**.
- This is a **miniature implementation**, and it **only predicts the dependencies**.
- The **UAS score** for evaluation is:
  - **Dev UAS:** 63.63% 
  - **Test UAS:** 64.54%
