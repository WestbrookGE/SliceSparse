#include <iostream>
#include <vector>
#include <omp.h> 
#include <map>

#include "utils.h"
#include "mat.h"
#include "timer.h"

using namespace std;

class sparse_operator
{
public:
	static double nrm2_sq( int siz, _float* x )
	{
		double ret = 0;
		for( int i=0; i<siz; i++ )
		{
			ret += SQ( x[i] );
		}
		return (ret);
	}

	static double dot( const double *s, int siz, _float* x )
	{
		double ret = 0;
		for( int i=0; i<siz; i++ )
		{
			ret += s[i] * x[i];
		}
		return (ret);
	}

	static void axpy(const double a, int siz, _float* x, double *y)
	{
		for( int i=0; i<siz; i++ )
		{
			y[i] += a * x[i];
		}
	}
};


void solve_l2r_lr_dual(_float** data,int num_trn, int num_ft,  int* y, double *w, double eps,	double Cp, double Cn, int svm_iter);
void solve_l2r_l1l2_svc(_float** data,int num_trn, int num_ft,  int* y, double *w, double eps,	double Cp, double Cn, int siter);
