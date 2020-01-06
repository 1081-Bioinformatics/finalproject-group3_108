# Title of your paper
### Members
* name, student ID1
* name, student ID2
* name, student ID3
* ...

### Demo
```bash
cd code

# Run preprocessing
make preprocess

# Run models
make lda lr svm rndfor elasticnet attention
```

## Folder organization and its related information

### docs
* Your presentation, 1081_bioinformatics_FP_< groupID >.ppt/pptx/pdf, by **01.07**
* Any related document for the paper
  * software user guide

### data
* Source
* Format
* Size

### code

#### Structure
- code/core  Core functions for most models
- code/preprocess  Preprocessing scripts
- code/util  Utility routines

#### Packages
- numpy (matrix utilities)
- scikit-learn (statistics routines)
- pytorch (attention model framework)
- keras (NN model framework)
- matplotlib (figure drawing)
- absl-py (flag management)
- coloredlogs (beautiful logging)

#### Analysis
- code/util/scorer.py
	Computes the AUC/CI for ROC curves

### results
#### Which part of the paper do you reproduce?
1. We reproduced several models (LDA, Logistic Regression, SVM, RandomForest, ElasticNet) in scikit-learn.
1. We also reproduced the NN model using Keras.
1. Additionally, we proposed a attention-baed model using PyTorch, and gained a good enough results. However, the results are still worse than the best models proposed by the original authors, maybe because of the lack of data complexity, witch cause our model easily to overfit the training data.

#### Any improvement or change by your package?
1. The original authors uses STATISTICA and Weka to produce their models. Since these programs either required commercial licenses or is hare to use, we chose to use python to reproduce the results. However, we are not able to reproduce the results since STATISTICA and Weka provides some fine-tuning that we don't know.

## References
* Packages you use
* Related publications
