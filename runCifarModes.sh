#!/usr/bin/env bash

RUNDIR=datasets/cifar_modes

dirs=${RUNDIR}/Cifar[1-9]
subdirs=`for d in $dirs; do  echo $d/source; done`
labels=`for d in $dirs; do  echo Q=${d#${RUNDIR}/Cifar}; done`
python prd_from_image_folders.py --reference_dir ${RUNDIR}/Cifar5/target --eval_dirs $subdirs --eval_labels $labels  --plot_path ./cifar_cluster.png --num_runs=10 --num_clusters 20 
python prd_from_image_folders.py --reference_dir ${RUNDIR}/Cifar5/target --eval_dirs $subdirs --eval_labels $labels  --plot_path ./cifar_classif.png --num_runs=10 --classif --num_epochs=50 --patience=10
