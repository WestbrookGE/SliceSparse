#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <random>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <omp.h> 
#include <sys/types.h> 
#include <sys/stat.h> 

#include "config.h"
#include "utils.h"
#include "mat.h"
#include "timer.h"
#include "svm.h"

using namespace std;

class Param
{
public:
	_int num_trn;
	_int num_ft;
	_int num_lbl;
	_int num_threads;
	_bool quiet;
	// HNSW Params
	_int M;
	_int efC;
	_int efS;
	_int num_nbrs;
	_int num_io_threads;
	// Discriminative Model Params
	_float classifier_cost;
	_float classifier_threshold;
	_int classifier_maxiter;
	_int classifier_kind;
	//Prediction param
	_float b_gen;

	Param()
	{
		num_trn = 0;
		num_ft = 0;
		num_lbl = 0;
		num_threads = 1;
		quiet = false;
		M = 100;
		efC = 300;
		efS = 300;
		num_nbrs = 300;
		num_io_threads = 20;
		classifier_cost = 1.0;
		classifier_threshold = 1e-6;
		classifier_maxiter = 20;
		classifier_kind = 0;
		b_gen = 0;
	}

	Param(string fname)
	{
		check_valid_filename(fname,true);
		ifstream fin;
		fin.open(fname);
		
		fin>>num_trn;
		fin>>num_ft;
		fin>>num_lbl;
		fin>>num_threads;
		fin>>quiet;
		fin>>M;
		fin>>efC;
		fin>>efS;
		fin>>num_nbrs;
		fin>>num_io_threads;
		fin>>classifier_cost;
		fin>>classifier_threshold;
		fin>>classifier_maxiter;
		fin>>classifier_kind;
		fin>>b_gen;
		fin.close();
	}

	void write(string fname)
	{
		check_valid_filename(fname,false);
		ofstream fout;
		fout.open(fname);

		fout<<num_trn<<"\n";
		fout<<num_ft<<"\n";
		fout<<num_lbl<<"\n";
		fout<<num_threads<<"\n";
		fout<<quiet<<"\n";
		fout<<M<<"\n";
		fout<<efC<<"\n";
		fout<<efS<<"\n";
		fout<<num_nbrs<<"\n";
		fout<<num_io_threads<<"\n";
		fout<<classifier_cost<<"\n";
		fout<<classifier_threshold<<"\n";
		fout<<classifier_maxiter<<"\n";
		fout<<classifier_kind<<"\n";
		fout<<b_gen<<"\n";
		fout.close();
	}
	void print()
	{
		cout<<"Number of training examples="<<num_trn<<"\n";
		cout<<"Number of features="<<num_ft<<"\n";
		cout<<"Number of labels="<<num_lbl<<"\n";
		cout<<"Number of train/test threads="<<num_threads<<"\n";
		cout<<"Quiet="<<quiet<<"\n";
		cout<<"M="<<M<<"\n";
		cout<<"efConstruction="<<efC<<"\n";
		cout<<"efSearch="<<efS<<"\n";
		cout<<"Number of nearest neighbors="<<num_nbrs<<"\n";
		cout<<"Number of threads for I/O="<<num_io_threads<<"\n";
		cout<<"Cost co-efficient for discriminative classifier="<<classifier_cost<<"\n";
		cout<<"Threshold for discriminative classifier="<<classifier_threshold<<"\n";
		cout<<"Maximum number of iterations for the discriminative classifier="<<classifier_maxiter<<"\n";
		cout<<"Separator Type="<<classifier_kind<<"\n";
		cout<<"b_gen="<<b_gen<<"\n";
	}
};

DMatF* train_slice(DMatF* trn_ft_mat, SMatF* trn_lbl_mat, string model_dir, Param& params, float& train_time);
SMatF* predict_slice(DMatF* tst_ft_mat, DMatF* w_dis, string model_dir, Param& params, float& test_time); 
