#!/bin/bash
set -e
dataset="$1"
n_neigh="$2"
n_thread="$3"

data_dir="/home/hzlb230017/data/$dataset"
model_dir="${data_dir}/model/${n_neigh}NN"
result_dir="${data_dir}/result"

mkdir -p $model_dir
mkdir -p $result_dir

trn_file="${data_dir}/train.txt"
tst_file="${data_dir}/test.txt"

trn_ft_file="${data_dir}/trn_ft_mat.txt"
trn_lbl_file="${data_dir}/trn_lbl_mat.txt"
tst_ft_file="${data_dir}/tst_ft_mat.txt"
tst_lbl_file="${data_dir}/tst_lbl_mat.txt"

score_file="${result_dir}/score${n_neigh}.txt"
time_file="${result_dir}/time${n_neigh}N${n_thread}T.txt"
eval_file="${result_dir}/eval${n_neigh}N.txt"
eval_sum_file="/home/hzlb230017/tmp/eval.txt"
time_sum_file="/home/hzlb230017/tmp/time.txt"
run_log_file="${result_dir}/log${n_neigh}N${n_thread}T.txt"

if [ ! -f $trn_ft_file ]; then
    perl /home/hzlb230017/code/Tree_Extreme_Classifiers/Tools/convert_format.pl $trn_file $trn_ft_file $trn_lbl_file
    perl /home/hzlb230017/code/Tree_Extreme_Classifiers/Tools/convert_format.pl $tst_file $tst_ft_file $tst_lbl_file
fi


echo "----------------Slice--------------------------"
trn_st=`date +%s`
./slice_train $trn_ft_file $trn_lbl_file $model_dir -m 100 -c 300 -s 300 -k $n_neigh -o $n_thread -t $n_thread -f 0.1 -siter 20 -b 2 -stype 0 -C 1 -q 0 | tee -a  $run_log_file
trn_et=`date +%s`
./slice_predict $tst_ft_file $model_dir $score_file | tee -a $run_log_file
tst_et=`date +%s`

echo "SparseSlice $dataset-$n_neigh"| tee -a  $eval_file $eval_sum_file 
./Tools/metrics/precision_k $score_file $tst_lbl_file 5 | tee -a  $eval_file $eval_sum_file 
./Tools/metrics/nDCG_k $score_file $tst_lbl_file 5 | tee -a  $eval_file $eval_sum_file 

echo $dataset "SparseSlice-trn $n_neigh $n_thread" `expr $trn_et - $trn_st` "seconds"| tee -a  $time_sum_file $time_file 
echo $dataset "SparseSlice-tst $n_neigh $n_thread" `expr $tst_et - $trn_et` "seconds"| tee -a  $time_sum_file $time_file 