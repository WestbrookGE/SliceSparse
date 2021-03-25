#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>

#include "config.h"
#include "utils.h"
#include "mat.h"

using namespace std;

void ndcg_k(SMatF* score_mat, SMatF* lbl_mat, int K)
{
	_int num_inst = score_mat->nc;
	_int num_lbl = lbl_mat->nr;

	float* ndcg = new float[K];
	for(int i=0;i<K;i++)
		ndcg[i]=0;
	for(_int i=0;i<num_inst;i++)
	{
		if (lbl_mat->size[i]==0)
			continue;

		std::map<_int, _int> lbl_map;
		float* den = new float[K];
		float* n = new float[K];
		for(int j=0;j<K;j++)
		{
			den[j]=0;
			n[j]=0.0;
		}
		for(_int j=0;j<lbl_mat->size[i];j++)
		{
			if(j==0)
				den[j] = 1;
			else if(j<K && j>0)
				den[j] = den[j-1] + log(2.0)/log(2+j);
			lbl_map[lbl_mat->data[i][j].first] = 1;
		}
		for(_int j=lbl_mat->size[i];j<K;j++)
			den[j] = den[lbl_mat->size[i]-1];
		pairIF* vec = score_mat->data[i];
		sort(vec, vec+score_mat->size[i], comp_pair_by_second_desc<_int,_float>);

		_int k = K;
		if (k>score_mat->size[i])
			k = score_mat->size[i];

		for(_int j=0;j<k;j++)
		{
			if(lbl_map.count(vec[j].first)>0)
			{
				for(_int pos=j;pos<K;pos++)
					n[pos] += log(2.0)/log(2+j);
			}
		}
		for(_int j=0; j<K; j++)
			ndcg[j] += n[j]/den[j];

	}
	for(int i=0;i<K;i++)
		printf("nDCG@%d = %f\n",i+1,ndcg[i]/((float)(num_inst)));
}

int main(int argc, char* argv[])
{
	string score_file = string( argv[1] );
	check_valid_filename( score_file, true );
	
	string lbl_file = string( argv[2] );
	check_valid_filename( lbl_file, true );
	
	int K = (int) stoi(string(argv[3]));

	
	SMatF* score_mat = new SMatF(score_file, 0);
	cout<<"score file read "<<score_file<<endl;
	
	SMatF* lbl_mat = new SMatF(lbl_file, 0);
	cout<<"lbl file read "<<lbl_file<<endl;

	cout<<"num_inst="<<score_mat->nc<<" num_lbl="<<lbl_mat->nr<<endl;
	assert(score_mat->nc==lbl_mat->nc);
	assert(score_mat->nr==lbl_mat->nr);
	ndcg_k(score_mat, lbl_mat, K);
	
	delete score_mat;
	delete lbl_mat;
}
