#include <iostream>
#include <fstream>
#include <string>

#include "timer.h"
#include "slice.h"

using namespace std;

void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./slice_train [feature file name] [label file name] [model dir name] -m 100 -c 300 -s 300 -k 300 -o 20 -t 1 -f 0.000001 -siter 20 -q 0"<<endl<<endl;

	cerr<<"-m HNSW M parameter. default=100"<<endl;
	cerr<<"-c HNSW efConstruction parameter. default=300"<<endl;
	cerr<<"-s HNSW efSearch parameter. default=300"<<endl;
	cerr<<"-k Number of labels to be shortlisted per training point according to the generative model. default=300"<<endl;
	cerr<<"-o Number of threads used to write the retrived ANN points to file. default=20"<<endl;
	cerr<<"-t Number of threads used to train ANNS datastructure and the discriminative classifiers. default=1"<<endl;
	cerr<<"-C SVM weight co-efficient. default=1.0"<<endl;
	cerr<<"-f Svm weights threshold. default=0.000001"<<endl;
	cerr<<"-siter Maximum iterations for training discriminative classifier. default=20"<<endl;
	cerr<<"-q quiet option (0/1). default=0"<<endl;
	cerr<<"-stype linear separator type. 0=L2 regularized squared hinge loss, 1=L2 regularized log loss. default=0"<<endl;
	cerr<<"feature file should be in dense matrix format and label file should be in sparse matrix format"<<endl;
	exit(1);
}

Param parse_param(_int argc, char* argv[])
{
	Param param;

	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt = string(argv[i]);
		sval = string(argv[i+1]);
		val = stof(sval);
		if(opt=="-m")
			param.M = (_int)val;
		else if(opt=="-c")
			param.efC = (_int)val;
		else if(opt=="-s")
			param.efS = (_int)val;
		else if(opt=="-k")
			param.num_nbrs = (_int)val;	
		else if(opt=="-o")
			param.num_io_threads = (_int)val;
		else if(opt=="-t")
			param.num_threads = (_int)val;
		else if(opt=="-C")
			param.classifier_cost = (_float)val;
		else if(opt=="-f")
			param.classifier_threshold = (_float)val;
		else if(opt=="-b")
			param.b_gen = (_float)val;
		else if(opt=="-siter")
			param.classifier_maxiter = (_int)val;
		else if(opt=="-q")
			param.quiet = (_bool)val;
		else if(opt=="-stype")
			param.classifier_kind = (_int)val;
	}
	return param;
}

SMatF* remove_labels_with_no_training_data(SMatF* trn_lbl_mat, string removed_lbl_file, Param& params)
{
	SMatF* tmat = trn_lbl_mat->transpose();
	vector<_int> removed_cols = tmat->remove_columns_with_no_data();
	if (!params.quiet)
		cout<<"Number of labels removed as they had no training data = "<<removed_cols.size()<<endl;
	if (removed_cols.size()==0)
		return trn_lbl_mat;

	SMatF* out_mat = tmat->transpose();

	ofstream fout;
	fout.open(removed_lbl_file);
	fout<<removed_cols.size()<<endl;
	for(_int i=0; i<removed_cols.size(); i++)
	{
		fout<<removed_cols[i]<<endl;
	}
	fout.close();

	delete tmat;
	return out_mat;
}
void remove_training_points_with_no_features(SMatF* trn_ft_mat, SMatF* trn_lbl_mat, Param& params)
{
	int ctr = 0;
	for(int i=0;i<trn_ft_mat->nc;i++)
	{
		float norm = 0;
		for(int j=0;j<trn_ft_mat->size[i];j++)
			norm += pow(trn_ft_mat->data[i][j].second, 2.0);
		if (norm==0)
		{
			delete [] trn_ft_mat->data[i];
			delete [] trn_lbl_mat->data[i];
			continue;
		}
		trn_ft_mat->data[ctr] = trn_ft_mat->data[i];
		trn_lbl_mat->data[ctr] = trn_lbl_mat->data[i];
		trn_lbl_mat->size[ctr] = trn_lbl_mat->size[i];
		ctr++;
	}
	if (!params.quiet)
		cout<<"Number of training points removed as they had no active features = "<<trn_ft_mat->nc-ctr<<endl;
	trn_ft_mat->nc = ctr;
	trn_lbl_mat->nc = ctr;
}

int main(int argc, char* argv[])
{

	string ft_file = string( argv[1] );
	check_valid_filename( ft_file, true );
	
	string lbl_file = string( argv[2] );
	check_valid_filename( lbl_file, true );
	
	string model_dir = string( argv[3] );
	check_valid_foldername( model_dir );

	Param params = parse_param( argc-4, argv+4 );

	SMatF* trn_ft_mat = new SMatF( ft_file, 0);
	SMatF* trn_lbl_mat = new SMatF(lbl_file, 0);


	remove_training_points_with_no_features(trn_ft_mat, trn_lbl_mat, params);
	SMatF* trn_lbl_mat_new = remove_labels_with_no_training_data(trn_lbl_mat, model_dir+"/no_data_labels.txt", params);


	params.num_trn = trn_ft_mat->nc;
	params.num_ft = trn_ft_mat->nr;
	params.num_lbl = trn_lbl_mat->nr;
	params.write( model_dir+"/param" );
	if (!params.quiet)
	{
		cout<<"Parameter Setting"<<endl;
		cout<<"-------------------------------------------------"<<endl;
		params.print();
		cout<<"-------------------------------------------------"<<endl;
	}

	if (!params.quiet)
		cout<<".............. Slice training started ..............."<<endl;
	_float train_time;
	SMatF* w_dis = train_slice(trn_ft_mat, trn_lbl_mat_new, model_dir, params, train_time);
	cout << "Total training time: " << train_time << " s" << endl;
	w_dis->write(model_dir+"/w_discriminative.txt", 0);
	delete trn_ft_mat;
	delete trn_lbl_mat;
	delete trn_lbl_mat_new;
	delete w_dis;
}
