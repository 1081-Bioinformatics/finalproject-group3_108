#!/bin/bash

IPREFIX=$(realpath -m $1)
ODIR=$2

mkdir -p ${ODIR}/train
ln -s ${IPREFIX}_x_train.npy ${ODIR}/train/x.npy
ln -s ${IPREFIX}_y_train.npy ${ODIR}/train/y.npy

mkdir -p ${ODIR}/test
ln -s ${IPREFIX}_x_test.npy ${ODIR}/test/x.npy
ln -s ${IPREFIX}_y_test.npy ${ODIR}/test/y.npy
