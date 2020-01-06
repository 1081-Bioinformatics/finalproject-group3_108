# Diagnostic potential for a serum miRNA neural network for detection of ovarian cancer
### Members
* 盧佳妤, 108753120
* 陳先灝, 108753107
* 段寶鈞, 108753116

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
* 1081_bioinformatics_FP_3.pptx/pdf, by **01.07**


### data
- Source
	- NCBI
		- [GEO Accession viewer (GSE94533)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94533)
		- [Run Selector (PRJNA371423)](https://www.ncbi.nlm.nih.gov/Traces/study/?acc=PRJNA371423)
	- EBI
		- [ENA Broswer (PRJNA371423)](https://www.ebi.ac.uk/ena/browser/view/PRJNA371423)
- Format
	- xlsx file
		- [GSE94533_Processed_file_Cohort1.xlsx](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE94533&format=file&file=GSE94533%5FProcessed%5Ffile%5FCohort1%2Exlsx%2Egz)
		- [GSE94533_Processed_file_Cohort2.xlsx](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE94533&format=file&file=GSE94533%5FProcessed%5Ffile%5FCohort2%2Exlsx%2Egz)
- Size
	- 4.89MB + 9.22MB



### code

#### Structure
- `code/core`
	- Core functions for most models
- `code/preprocess`
	- Preprocessing scripts
- `code/util`
	- Utility routines

#### Packages
- numpy (matrix utilities)
- scikit-learn (statistics routines)
- pytorch (attention model framework)
- keras (NN model framework)
- matplotlib (figure drawing)
- absl-py (flag management)
- coloredlogs (beautiful logging)

#### Analysis
- `code/util/scorer.py`
	- Computes the AUC/CI for ROC curves

### results
#### Which part of the paper do you reproduce?
1. We reproduced several models (LDA, Logistic Regression, SVM, RandomForest, ElasticNet) in scikit-learn.
1. We also reproduced the NN model using Keras.
1. Additionally, we proposed a attention-baed model using PyTorch, and gained a good enough results. However, the results are still worse than the best models proposed by the original authors, maybe because of the lack of data complexity, witch cause our model easily to overfit the training data.

#### Any improvement or change by your package?
1. The original authors uses STATISTICA and Weka to produce their models. Since these programs either required commercial licenses or is hare to use, we chose to use python to reproduce the results. However, we are not able to reproduce the results since STATISTICA and Weka provides some fine-tuning that we don't know.

## References
* Packages we use
	- numpy 
	- scikit-learn 
	- pytorch 
	- keras
	- matplotlib 
	- absl-py
	- coloredlogs
