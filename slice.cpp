#include "slice.h"

using namespace std;
thread_local mt19937 reng; // random number generator used during training 

_int get_rand_num( _int siz )
{
	_llint r = reng();
	_int ans = r % siz;
	return ans;
}


#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)
typedef signed char schar;

void solve_l2r_lr_dual(_float** data,int num_trn, int num_ft,  int* y, double *w, double eps,	double Cp, double Cn, int classifier_maxiter)
{
	int l = num_trn;
	int w_size = num_ft;
	int i, s, iter = 0;

	double *xTx = new double[l];
	int max_iter = classifier_maxiter;
	int *index = new int[l];	
	double *alpha = new double[2*l]; // store alpha and C - alpha
	int max_inner_iter = 100; // for inner Newton
	double innereps = 1e-2;
	double innereps_min = min(1e-8, (double)eps);
	double upper_bound[3] = {Cn, 0, Cp};

	// Initial alpha can be set here. Note that
	// 0 < alpha[i] < upper_bound[GETI(i)]
	// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;

	for(i=0; i<l; i++)
	{
		xTx[i] = sparse_operator::nrm2_sq( num_ft, data[i] );
		sparse_operator::axpy(y[i]*alpha[2*i], num_ft, data[i], w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i=0; i<l; i++)
		{
			int j = i + get_rand_num( l-i );
			swap(index[i], index[j]);
		}

		int newton_iter = 0;
		double Gmax = 0;
		for (s=0; s<l; s++)
		{
			i = index[s];
			const _int yi = y[i];
			double C = upper_bound[GETI(i)];
			double ywTx = 0, xisq = xTx[i];
			ywTx = yi*sparse_operator::dot( w, num_ft, data[i] );
			double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
			if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
			{
				ind1 = 2*i+1;
				ind2 = 2*i;
				sign = -1;
			}

			//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
			double alpha_old = alpha[ind1];
			double z = alpha_old;
			if(C - z < 0.5 * C)
				z = 0.1*z;
			double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const _double eta = 0.1; // xi in the paper
			int inner_iter = 0;
			while (inner_iter <= max_inner_iter)
			{
				if(fabs(gp) < innereps)
					break;
				double gpp = a + C/(C-z)/z;
				double tmpz = z - gp/gpp;
				if(tmpz <= 0)
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				newton_iter++;
				inner_iter++;
			}

			if(inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C-z;
				sparse_operator::axpy(sign*(z-alpha_old)*yi, num_ft, data[i], w);
			}
		}

		iter++;

		if(Gmax < eps)
			break;

		if(newton_iter <= l/10)
			innereps = max(innereps_min, 0.1*innereps);

	}

	delete [] xTx;
	delete [] alpha;
	delete [] index;
}


void solve_l2r_l1l2_svc(_float** data,int num_trn, int num_ft,  int* y, double *w, double eps,	double Cp, double Cn, int siter)
{
	int l = num_trn;
	int w_size = num_ft;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = siter;
	int *index = new int[l];
	double *alpha = new double[l];
	int active_size = l;

	int tot_iter = 0;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};

	//d = pwd;
	//Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]

	
	for(i=0; i<l; i++)
		alpha[i] = 0;
	

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		//feature_node * const xi = prob->x[i];
		QD[i] += sparse_operator::nrm2_sq( num_ft, data[i] );
		sparse_operator::axpy(y[i]*alpha[i], num_ft, data[i], w);

		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			tot_iter ++;

			i = index[s];
			const int yi = y[i];
			//feature_node * const xi = prob->x[i];

			G = yi*sparse_operator::dot( w, num_ft, data[i] )-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			//cout << "update: " << i << " " << fabs(PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				//cout << " " << d;

				//print_nnz( w_size, w );

				sparse_operator::axpy(d, num_ft, data[i], w);

				//print_nnz( w_size, w );
			}
			//cout << endl;
		}

		iter++;

		/*
		if(iter % 10 == 0)
			cout << "." << endl;
		*/

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				//cout << "*" << endl;
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	//cout << " " << iter << " " << tot_iter ;

	// calculate objective value

	delete [] QD;
	delete [] alpha;
	delete [] index;
}
	

SMatF* compute_mu_plus(SMatF* trn_ft_mat, SMatF* trn_lbl_mat)
{
	SMatF* trn_lbl_mat_trans = trn_lbl_mat->transpose();
	SMatF* mu_plus = trn_ft_mat->prod(trn_lbl_mat_trans);
	mu_plus->unit_normalize_columns();
	return mu_plus;
}

void train_slice_generative_model(SMatF* trn_ft_mat, SMatF* trn_lbl_mat, string model_dir, string temp_dir, Param& params)
{
	SMatF* mu_plus = compute_mu_plus(trn_ft_mat, trn_lbl_mat);
	string mu_file = temp_dir+"/mu_plus.bin";
	mu_plus->write(mu_file, 1);

	string command = "python ANNS/train_hnsw.py ";
	string arguments =  mu_file + " " + model_dir + "/anns_model " + to_string(params.M) + " " + to_string(params.efC) + " " + to_string(params.num_threads) + " " + to_string(params.num_ft) + " cosinesimil";
	command += arguments;
	system(command.c_str());
}

IMat* read_multiple_imat_files(string file_dir, int num_files, bool input_format_is_binary)
{
	IMat* imat = new IMat(file_dir+"/0", input_format_is_binary);
	for(int i=1; i<num_files; i++)
	{
		IMat* tmat = new IMat(file_dir+"/"+to_string(i), input_format_is_binary);
		imat->append_mat_columnwise(tmat);
	}
	return imat;
}

IMat* find_most_confusing_negatives(SMatF* trn_ft_mat, string model_dir, string temp_dir, Param& params, float& io_time)
{
	Timer timer;
	timer.start();
	string trn_ft_file = temp_dir+"/trn_ft_mat.bin";
	trn_ft_mat->write(trn_ft_file, 1);
	timer.stop();
	string command = "python ANNS/test_hnsw.py ";
	string arguments = trn_ft_file + " " + model_dir + "/anns_model " + to_string(params.num_ft) + " " + to_string(params.num_lbl) + " " + to_string(params.efS) + " " + to_string(params.num_nbrs) + " 0 " + temp_dir + " " + to_string(params.num_threads) + " " + to_string(params.num_io_threads) + " cosinesimil";
	command += arguments;
  system(command.c_str());
	timer.resume();
	IMat* temp_imat = read_multiple_imat_files(temp_dir, params.num_io_threads, 0);
	IMat* trn_negatives = temp_imat->transpose();
	io_time = timer.stop();
	delete temp_imat;
	return trn_negatives;
}	

SMatF* train_discriminative_classifier(SMatF* trn_ft_mat, SMatF* trn_lbl_mat, IMat* trn_negatives, Param& params)
{
	SMatF* trn_lbl_mat_trans = trn_lbl_mat->transpose();
	trn_ft_mat->unit_normalize_columns();
	_int num_ft = trn_ft_mat->nr;
	for(_int i=0; i<trn_ft_mat->nc; i++)
	{
		Realloc(num_ft,num_ft+1,trn_ft_mat->data[i]);
		trn_ft_mat->data[i][num_ft] = make_pair( trn_ft_mat->nr, 1 );
	}
	trn_ft_mat->nr = num_ft+1;	
	_int num_trn = trn_ft_mat->nc;
	num_ft = trn_ft_mat->nr;
	_int num_lbl = trn_lbl_mat_trans->nc;

	float th = params.classifier_threshold;
	double eps = 0.1;
	double Cp = params.classifier_cost;
	double Cn = params.classifier_cost; 

	SMatF* w_dis = new SMatF( num_ft, num_lbl );

	omp_set_dynamic(0);
	omp_set_num_threads(params.num_threads);
	#pragma omp parallel shared(trn_ft_mat,trn_lbl_mat_trans,trn_negatives, w_dis, num_ft, num_lbl, th, eps, Cp, Cn)
	{
	#pragma omp for
	for( int l=0; l<num_lbl; l++)
	{
		_int sl_size = 0;
		VecI positives(num_trn, 0);
		for (int i=0; i<trn_lbl_mat_trans->size[l]; i++)
			positives[trn_lbl_mat_trans->data[l][i].first] = 1;

		int overlap = 0;
		for (int i=0; i<trn_negatives->size[l]; i++)
		{
			if (positives[trn_negatives->data[l][i]]>0)
			{
				positives[trn_negatives->data[l][i]] = -1;
				overlap++;
			}
		}
		sl_size = trn_negatives->size[l]+trn_lbl_mat_trans->size[l]-overlap;
		/* 
		if (sl_size==0)
		{
			for( _int f=0; f<num_ft; f++ )
				w_dis->data[l][f] = 0;
			continue;
		} */
		
		_int* y = new _int[sl_size];
		SMatF* data = new SMatF(num_ft, sl_size);
		_int inst;
		for (int i=0; i<trn_negatives->size[l]; i++)
		{
			inst = trn_negatives->data[l][i];
			data->data[i] = trn_ft_mat->data[inst];
			if (positives[inst]!=0)
				y[i] = +1;
			else
				y[i] = -1;
		}
		int ctr = trn_negatives->size[l];
		for (int i=0; i<trn_lbl_mat_trans->size[l]; i++)
		{
			inst = trn_lbl_mat_trans->data[l][i].first;
			if (positives[inst]>0)
			{
				data->data[ctr] = trn_ft_mat->data[inst];
				y[ctr] = +1;
				ctr++;
			}
		}
		
		float* w = new float[ num_ft ]();		
		if(params.classifier_kind==0)
			//solve_l2r_l1l2_svc( data, sl_size, num_ft, y, w, eps, Cp, Cn, params.classifier_maxiter);
			solve_l2r_l1l2_svc_sparse(data, y, w, eps, Cp, Cn, params.classifier_maxiter);
		else
			solve_l2r_lr_dual_sparse(data, y, w, eps, Cp, Cn, params.classifier_maxiter);
		_int siz = 0;
		for( _int f=0; f<num_ft; f++ )
		{
			if( fabs( w[f] ) > th )
				w_dis->data[ l ][ siz++ ] = make_pair( f, w[f] );
		}
		Realloc( num_ft, siz, w_dis->data[ l ] );
		w_dis->size[ l ] = siz;

		delete [] y;
		delete [] data;
		delete [] w;
	}
	}
	
	delete trn_lbl_mat_trans;
	return w_dis;
}
SMatF* train_slice(SMatF* trn_ft_mat, SMatF* trn_lbl_mat, string model_dir, Param& params, float& train_time)
{
	float io_time = 0.0;
	float* t_time = new float;
	*t_time = 0;
	Timer timer;

	string temp_dir = model_dir + "/tmp";
	mkdir(temp_dir.c_str(), S_IRWXU);
	timer.start();

	if (!params.quiet)
		printf("Training generative model ...\n");	
	train_slice_generative_model(trn_ft_mat, trn_lbl_mat, model_dir, temp_dir, params);

	if (!params.quiet)
		printf("Finding the most confusing negatives ...\n");
	IMat* trn_negatives = find_most_confusing_negatives(trn_ft_mat, model_dir, temp_dir, params, io_time);

	if (!params.quiet)
		printf("Training discriminative classifiers ...\n");
	SMatF* w_discriminative = train_discriminative_classifier(trn_ft_mat, trn_lbl_mat, trn_negatives, params);

	*t_time += timer.stop();
	train_time = *t_time;
	train_time -= io_time;
	delete t_time;
	delete trn_negatives;
	return w_discriminative;
}

SMatF* evaluate_discriminative_model(DMatF* tst_ft_mat, DMatF* w_dis, SMatF* shortlist, Param& params, int K)
{
	_int num_ft = tst_ft_mat->nr;
	_int num_lbl = w_dis->nc;
	_int num_tst = tst_ft_mat->nc;

	float gamma = 0;
	for (int i=0;i<num_lbl;i++)
	{
		float temp = 0;	
		for (int j=0;j<num_ft;j++)
			temp += pow(w_dis->data[i][j],2.0);
		gamma += sqrt(temp);
	}
	gamma = gamma/float(num_lbl);

	SMatF* score_mat = new SMatF();	
	score_mat->nr = num_lbl;
	score_mat->nc = num_tst;
	score_mat->size = new _int[num_tst]();
	score_mat->data = new pairIF*[num_tst];

	omp_set_dynamic(0);
	omp_set_num_threads(params.num_threads);
	#pragma omp parallel shared(tst_ft_mat,w_dis,shortlist,score_mat,num_ft, num_tst)
	{
	#pragma omp for
	for(_int i=0; i<num_tst; i++)
	{
		//if ((i%1000)==0)
		//	printf("%d\n",i);
		
		score_mat->data[i] = new pairIF[shortlist->size[i]];
		_int ctr = 0;
		_float max1 = 0;
		_float max_dist = 0;
		for(_int j=0; j<shortlist->size[i]; j++)
		{
			_int ind = shortlist->data[i][j].first;
			_float prod = 0;
			_int ctr1 = 0;
			_int ctr2 = 0;
			for(_int f=0; f<num_ft; f++)
				prod += w_dis->data[ind][f]*tst_ft_mat->data[i][f];
			score_mat->data[i][j].first = ind;
			score_mat->data[i][j].second = prod;
		}
		for(_int j=0; j<shortlist->size[i]; j++)
			score_mat->data[i][j].second = (1.0/(1.0+exp(-score_mat->data[i][j].second))) + (1.0/(1.0 + exp(-shortlist->data[i][j].second*gamma + params.b_gen)));
		pairIF* vec = score_mat->data[i];
		sort(vec, vec+shortlist->size[i], comp_pair_by_second_desc<_int,_float>);
		_int k = K;
		if (k>shortlist->size[i])
			k = shortlist->size[i];
		Realloc(shortlist->size[i],k,score_mat->data[i]);
		vec = score_mat->data[i];
		sort(vec, vec+k, comp_pair_by_first<_int,_float>);
		score_mat->size[i] = k;
	}
	}
	return score_mat;
}

SMatF* read_multiple_smat_files(string file_dir, int num_files, bool input_format_is_binary)
{
	SMatF* smat = new SMatF(file_dir+"/0", input_format_is_binary);
	for(int i=1; i<num_files; i++)
	{
		SMatF* tmat = new SMatF(file_dir+"/"+to_string(i), input_format_is_binary);
		smat->append_mat_columnwise(tmat);
	}
	return smat;
}


SMatF* evaluate_generative_model(DMatF* tst_ft_mat, string model_dir, string temp_dir, Param& params, float& io_time)
{
	Timer timer;
	timer.start();
	string tst_ft_file = temp_dir+"/tst_ft_mat.bin";
	tst_ft_mat->write(tst_ft_file, 1);
	timer.stop();
	string command = "python ANNS/test_hnsw.py ";
	string arguments = tst_ft_file + " " + model_dir + "/anns_model " + to_string(params.num_ft) + " " + to_string(params.num_lbl) + " " + to_string(params.efS) + " " + to_string(params.num_nbrs) + " 1 " + temp_dir + " " + to_string(params.num_threads) + " " + to_string(params.num_io_threads) + " cosinesimil";
	command += arguments;
  system(command.c_str());
	timer.resume();
	SMatF* generative_score_mat = read_multiple_smat_files(temp_dir, params.num_io_threads, 0);
	io_time = timer.stop();
	return generative_score_mat;
}	

SMatF* predict_slice(DMatF* tst_ft_mat, DMatF* w_dis, string model_dir, Param& params, float& test_time) 
{
	float io_time = 0.0;
	float* t_time = new float;
	*t_time = 0;
	Timer timer;

	string temp_dir = model_dir + "/tmp";
	mkdir(temp_dir.c_str(), S_IRWXU);

	timer.start();
	tst_ft_mat->unit_normalize_columns();
	SMatF* score_mat_gen = evaluate_generative_model(tst_ft_mat, model_dir, temp_dir, params, io_time);
	
	_int num_ft = tst_ft_mat->nr;
	for(_int i=0; i<tst_ft_mat->nc; i++)
	{
		Realloc(num_ft,num_ft+1,tst_ft_mat->data[i]);
		tst_ft_mat->data[i][num_ft] = 1;
	}
	tst_ft_mat->nr = num_ft+1;
	
	SMatF* score_mat = evaluate_discriminative_model(tst_ft_mat, w_dis, score_mat_gen, params, 20);
	*t_time += timer.stop();
	test_time = *t_time;
	test_time -= io_time;

	delete score_mat_gen;
	return score_mat;
}

void solve_l2r_l1l2_svc_sparse( SMatF* X_Xf, _int* y, _float *w, _float eps, _float Cp, _float Cn, _int classifier_maxitr )
{
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;

	_int i, s, iter = 0;
	_float C, d, G;
	_float *QD = new _float[l];
	_int max_iter = classifier_maxitr;
	_int *index = new _int[l];
	_float *alpha = new _float[l];
	_int active_size = l;

	_int tot_iter = 0;

	// PG: projected gradient, for shrinking and stopping
	_float PG;
	_float PGmax_old = INF;
	_float PGmin_old = -INF;
	_float PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	_float diag[3] = {(_float)0.5/Cn, (_float)0, (_float)0.5/Cp};
	_float upper_bound[3] = {INF, 0, INF};

	_int* size = X_Xf->size;
	pairIF** data = X_Xf->data;

	//d = pwd;
	//Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]

	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;

	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];
		QD[i] += sparse_operator::nrm2_sq( size[i], data[i] );
		sparse_operator::axpy(y[i]*alpha[i], size[i], data[i], w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			_int j = i + get_rand_num( active_size-i );
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			tot_iter ++;

			i = index[s];
			const _int yi = y[i];

			G = yi*sparse_operator::dot( w, size[i], data[i] )-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				_float alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], (_float)0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				sparse_operator::axpy(d, size[i], data[i], w);
			}
		}

		iter++;

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	// calculate objective value

	delete [] QD;
	delete [] alpha;
	delete [] index;
}

void solve_l2r_lr_dual_sparse( SMatF* X_Xf, _int* y, _float *w, _float eps, _float Cp, _float Cn, _int classifier_maxitr )
{
	_int l = X_Xf->nc;
	_int w_size = X_Xf->nr;
	_int i, s, iter = 0;

	_double *xTx = new _double[l];
	_int max_iter = classifier_maxitr;
	_int *index = new _int[l];	
	_double *alpha = new _double[2*l]; // store alpha and C - alpha
	_int max_inner_iter = 100; // for inner Newton
	_double innereps = 1e-2;
	_double innereps_min = min(1e-8, (_double)eps);
	_double upper_bound[3] = {Cn, 0, Cp};

	_int* size = X_Xf->size;
	pairIF** data = X_Xf->data;

	// Initial alpha can be set here. Note that
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;

	for(i=0; i<l; i++)
	{
		xTx[i] = sparse_operator::nrm2_sq( size[i], data[i] );
		sparse_operator::axpy(y[i]*alpha[2*i], size[i], data[i], w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i=0; i<l; i++)
		{
			_int j = i + get_rand_num( l-i );
			swap(index[i], index[j]);
		}

		_int newton_iter = 0;
		_double Gmax = 0;
		for (s=0; s<l; s++)
		{
			i = index[s];
			const _int yi = y[i];
			_double C = upper_bound[GETI(i)];
			_double ywTx = 0, xisq = xTx[i];
			ywTx = yi*sparse_operator::dot( w, size[i], data[i] );
			_double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			_int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
			if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
			{
				ind1 = 2*i+1;
				ind2 = 2*i;
				sign = -1;
			}

			_double alpha_old = alpha[ind1];
			_double z = alpha_old;
			if(C - z < 0.5 * C)
				z = 0.1*z;
			_double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const _double eta = 0.1; // xi in the paper
			_int inner_iter = 0;
			while (inner_iter <= max_inner_iter)
			{
				if(fabs(gp) < innereps)
					break;
				_double gpp = a + C/(C-z)/z;
				_double tmpz = z - gp/gpp;
				if(tmpz <= 0)
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				newton_iter++;
				inner_iter++;
			}

			if(inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C-z;
				sparse_operator::axpy(sign*(z-alpha_old)*yi, size[i], data[i], w);
			}
		}

		iter++;

		if(Gmax < eps)
			break;

		if(newton_iter <= l/10)
			innereps = max(innereps_min, 0.1*innereps);

	}

	delete [] xTx;
	delete [] alpha;
	delete [] index;
}