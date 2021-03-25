#include <iostream>
#include <fstream>
#include <string>

#include "timer.h"
#include "slice.h"

using namespace std;

void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./slice_predict [feature file name] [label file name] [model dir name] -t 1 -b 0 -q 0"<<endl<<endl;

	cerr<<"-s HNSW efSearch parameter. default=300"<<endl;
	cerr<<"-k Number of labels to be shortlisted per test point according to the generative model. default=300"<<endl;
	cerr<<"-o Number of threads used to write the retrived ANN points to file. default=20"<<endl;
	cerr<<"-t Number of threads for evaluating discriminative and generative models. default=1"<<endl;
	cerr<<"-b Bias parameter for the generative model. default=0"<<endl;
	cerr<<"-q quiet option (0/1). default=0"<<endl;
	cerr<<"feature file should be in dense matrix format"<<endl;
	exit(1);
}

Param parse_param(int argc, char* argv[], string model_dir)
{
	Param param(model_dir+"/param");

	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt = string(argv[i]);
		sval = string(argv[i+1]);
		val = stof(sval);
		if(opt=="-s")
			param.efS = (_int)val;
		else if(opt=="-b")
			param.b_gen = (_float)val;	
		else if(opt=="-k")
			param.num_nbrs = (_int)val;	
		else if(opt=="-o")
			param.num_io_threads = (_int)val;
		else if(opt=="-t")
			param.num_threads = (_int)val;
		else if(opt=="-q")
			param.quiet = (_bool)val;
	}

	return param;
}
SMatF* add_empty_rows(SMatF* smat, string removed_rows_file)
{
	_int num_rem_rows;
	ifstream fin;
	fin.open(removed_rows_file);
	fin>>num_rem_rows;
	if (num_rem_rows == 0)
	{
		fin.close();
		return smat;
	}
	
	vector<_int> removed_rows;
	_int r;
	for(_int i=0; i<num_rem_rows; i++)
	{
		fin>>r;
		removed_rows.push_back(r);
	}
	fin.close();

	_int nr = smat->nr+num_rem_rows;
	std::map<_int, _int> row_map;
	_int ctr1 = 0;
	_int ctr2 = 0;
	for(_int i=0;i<nr;i++)	
	{
		if (ctr1<num_rem_rows && removed_rows[ctr1]==i)
		{
			ctr1++;
			continue;
		}
		row_map[ctr2] = i;
		ctr2++;
	}
	for(_int i=0; i<smat->nc; i++)
	{
		for(_int j=0;j<smat->size[i];j++)
			smat->data[i][j].first = row_map[smat->data[i][j].first];
	}
	smat->nr = nr;
	return smat;
}


int main(int argc, char* argv[])
{

	string ft_file = string( argv[1] );
	check_valid_filename( ft_file, true );
	
	string model_dir = string( argv[2] );
	check_valid_foldername( model_dir );

	string out_file = string( argv[3] );
	check_valid_filename( out_file, false );
	
	Param params = parse_param( argc-4, argv+4, model_dir);

	DMatF* tst_ft_mat = new DMatF( ft_file, 0);
	DMatF* w_dis = new DMatF(model_dir+"/w_discriminative.txt", 0);

	if (!params.quiet)
		cout<<".............. Slice prediction started ..............."<<endl;

	_float test_time;
	SMatF* score_mat = predict_slice(tst_ft_mat, w_dis, model_dir, params, test_time);
	cout << "Total prediction time: " << test_time << " s" << endl;
	cout << "Prediction time per point: " << test_time*1000/(float)tst_ft_mat->nc << " ms" << endl;

	string removed_labels_file = model_dir+"/no_data_labels.txt";
	ifstream fin(removed_labels_file);
	if (fin.good())
		score_mat = add_empty_rows(score_mat, removed_labels_file);

	score_mat->write(out_file, 0);
	delete tst_ft_mat;
	delete score_mat;
}
