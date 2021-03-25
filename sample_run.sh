#!/bin/bash
set -e

dataset="EURLex-4K"
data_dir="/home/hzlb230017/data/$dataset"
results_dir="/home/hzlb230017/data/$dataset/SliceSparseResult"
model_dir="/home/hzlb230017/data/$dataset/model"
mkdir -p $model_dir
mkdir -p $results_dir

trn_ft_file_sparse="${data_dir}/trn_ft_mat.txt"
tst_ft_file_sparse="${data_dir}/tst_ft_mat.txt"

trn_ft_file="${data_dir}/trn_ft_mat_dense.txt"
trn_lbl_file="${data_dir}/trn_lbl_mat.txt"
tst_ft_file="${data_dir}/tst_ft_mat_dense.txt"
tst_lbl_file="${data_dir}/tst_lbl_mat.txt"
score_file="${results_dir}/score_mat.txt"


echo "Converting sparse feature matrices to dense format"
./Tools/c++/smat_to_dmat $trn_ft_file_sparse $trn_ft_file
./Tools/c++/smat_to_dmat $tst_ft_file_sparse $tst_ft_file


echo "----------------Slice--------------------------"
./slice_train $trn_ft_file $trn_lbl_file $model_dir -m 100 -c 300 -s 300 -k 300 -o 20 -t 1 -f 0.000001 -siter 20 -b 2 -stype 0 -C 1 -q 0
./slice_predict $tst_ft_file $model_dir $score_file
./Tools/metrics/precision_k $score_file $tst_lbl_file 5
./Tools/metrics/nDCG_k $score_file $tst_lbl_file 5

