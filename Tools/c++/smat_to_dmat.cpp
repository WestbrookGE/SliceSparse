#include <iostream>
#include <string>
#include <omp.h> 

#include "config.h"
#include "utils.h"  
#include "mat.h" 

inline void smat_to_dmat(string outfile, string fname)
{
	ofstream fout;
	ifstream fin;
	_int nc, nr;
	fin.open(fname);		
	fin>>nc>>nr;
	fout.open(outfile);
	fout<<nc<<" "<<nr<<endl;
	fin.ignore();
	for(_int i=0; i<nc; i++)
	{
		string line;
		getline(fin,line);
		line += "\n";
		_int pos = 0;
		_int next_pos;
		_int ind = 0;
		vector<_float> vals(nr,0);
		while(next_pos=line.find_first_of(": \n",pos))
		{
			if((size_t)next_pos==string::npos)
				break;
			ind = (_int) stoi(line.substr(pos,next_pos-pos));
			pos = next_pos+1;

			next_pos = line.find_first_of(": \n",pos);
			if((size_t)next_pos==string::npos)
				break;
			
			vals[ind] = (_float)stof(line.substr(pos,next_pos-pos));
			pos = next_pos+1;
		}
		for(_int j=0; j<nr; j++)
		{
			if(j==0)
				fout<<vals[j];
			else
				fout<<" "<<vals[j];
		}
		fout<<endl;
	}	
	fin.close();
	fout.close();	
}

int main(int argc, char* argv[])
{
	string mat_file = string(argv[1]);
	check_valid_filename(mat_file, true);
	string out_file = string(argv[2]);
	check_valid_filename(out_file, false);
	smat_to_dmat(out_file, mat_file);
}
