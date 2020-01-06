PY = python3 -m

.PHONY: all run preprocess

all: run

run:
	make preprocess
	make train

preprocess:
	$(PY) preprocess.mirna2npy
	$(PY) preprocess.select_feature --flagfile ../data/cfg/preprocess/1_significance.cfg
	$(PY) preprocess.select_feature --flagfile ../data/cfg/preprocess/2_correlation.cfg
	$(PY) preprocess.select_feature --flagfile ../data/cfg/preprocess/3_fold.cfg

train:
	$(PY) core.main --model_name=lda --input_name=0_original
	$(PY) core.main --model_name=lda --input_name=1_significance
	$(PY) core.main --model_name=lda --input_name=2_correlation
	$(PY) core.main --model_name=lda --input_name=3_fold

	$(PY) core.main --model_name=lr --input_name=0_original
	$(PY) core.main --model_name=lr --input_name=1_significance
	$(PY) core.main --model_name=lr --input_name=2_correlation
	$(PY) core.main --model_name=lr --input_name=3_fold

	$(PY) core.main --model_name=svm --input_name=0_original
	$(PY) core.main --model_name=svm --input_name=1_significance
	$(PY) core.main --model_name=svm --input_name=2_correlation
	$(PY) core.main --model_name=svm --input_name=3_fold

	$(PY) core.main --model_name=nb --input_name=0_original
	$(PY) core.main --model_name=nb --input_name=1_significance
	$(PY) core.main --model_name=nb --input_name=2_correlation
	$(PY) core.main --model_name=nb --input_name=3_fold

	$(PY) core.main --model_name=ft --input_name=0_original
	$(PY) core.main --model_name=ft --input_name=1_significance
	$(PY) core.main --model_name=ft --input_name=2_correlation
	$(PY) core.main --model_name=ft --input_name=3_fold

	$(PY) core.main --model_name=rndfor --input_name=0_original
	$(PY) core.main --model_name=rndfor --input_name=1_significance
	$(PY) core.main --model_name=rndfor --input_name=2_correlation
	$(PY) core.main --model_name=rndfor --input_name=3_fold

	$(PY) core.main --model_name=elasticnet --input_name=0_original
	$(PY) core.main --model_name=elasticnet --input_name=1_significance
	$(PY) core.main --model_name=elasticnet --input_name=2_correlation
	$(PY) core.main --model_name=elasticnet --input_name=3_fold

	$(PY) core.main --model_name=attention --input_name=0_original
	$(PY) core.main --model_name=attention --input_name=1_significance
	$(PY) core.main --model_name=attention --input_name=2_correlation
	$(PY) core.main --model_name=attention --input_name=3_fold
