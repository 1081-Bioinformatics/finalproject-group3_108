PY = python3 -m
SH = bash -x

DATADIR = ../data
MODELS = lda lr svm rndfor elasticnet attention

.PHONY: all run preprocess train $(MODELS)

all: run

run:
	make preprocess
	make train

preprocess:
	$(PY) preprocess.mirna2npy
	$(PY) preprocess.select_feature --flagfile ../data/cfg/preprocess/1_significance.cfg
	$(PY) preprocess.select_feature --flagfile ../data/cfg/preprocess/2_correlation.cfg
	$(PY) preprocess.select_feature --flagfile ../data/cfg/preprocess/3_fold.cfg
	$(SH) ./preprocess/link_data.sh $(DATADIR)/control/control $(DATADIR)/npy/control
	$(SH) ./preprocess/link_data.sh $(DATADIR)/log2_change/log2_fold_change_filter $(DATADIR)/npy/log2_fold
	$(SH) ./preprocess/link_data.sh $(DATADIR)/simple_filter/simple_fold_change_filter $(DATADIR)/npy/simple_fold

train: $(MODELS)

$(MODELS):
	$(PY) core.main --model_name=$@ --input_name=0_original
	$(PY) core.main --model_name=$@ --input_name=1_significance
	$(PY) core.main --model_name=$@ --input_name=2_correlation
	$(PY) core.main --model_name=$@ --input_name=3_fold
	$(PY) core.main --model_name=$@ --input_name=control
	$(PY) core.main --model_name=$@ --input_name=log2_fold
	$(PY) core.main --model_name=$@ --input_name=simple_fold
