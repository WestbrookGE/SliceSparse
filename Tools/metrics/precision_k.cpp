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

void precision_k(SMatF* score_mat, SMatF* lbl_mat, int K)
{
	_int num_inst = score_mat->nc;
	_int num_lbl = lbl_mat->nr;
	int* p = new int[K];
	for(int i=0;i<K;i++)
		p[i]=0;
	for(_int i=0;i<num_inst;i++)
	{
		std::map<_int, _int> lbl_map;
		for(_int j=0;j<lbl_mat->size[i];j++)
			lbl_map[lbl_mat->data[i][j].first] = 1;
			
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
					p[pos]++;
			}
		}
	}
	for(int i=0;i<K;i++)
		printf("Precision@%d = %f\n",i+1,(float)p[i]/((float)(num_inst*(i+1))));
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
	precision_k(score_mat, lbl_mat, K);
	
	delete score_mat;
	delete lbl_mat;
}
