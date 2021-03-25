#include <iostream>
#include <string>
#include <omp.h> 

#include "config.h"
#include "utils.h"  
#include "mat.h" 

inline void dmat_to_smat(string outfile, string fname)
{
	ofstream fout;	
	ifstream fin;
	_int nc, nr;

	fin.open(fname);
	fin>>nc>>nr;

	fout.open(outfile);
	fout<<nc<<" "<<nr<<endl;
		
	vector<_float> vals;
	fin.ignore();
	for (_int column = 0; column < (nc); ++column) 
	{
		vals.clear();
		string line;
		getline(fin,line);
		line += "\n";
		_int pos = 0;
		_int next_pos;

		while(next_pos=line.find_first_of(" \n",pos))
		{
			if((size_t)next_pos==string::npos)
				break;
				
			vals.push_back(stof(line.substr(pos,next_pos-pos)));
			pos = next_pos+1;
		}

		assert(vals.size()==nr);
		int flag =0;
		for (_int row = 0; row < nr; ++row) 
		{
			if(vals[row]==0)
				continue;
			if(flag==0)
			{
				flag=1;
				fout<<row<<":"<<vals[row];
			}
			else
				fout<<" "<<row<<":"<<vals[row];
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
	
	dmat_to_smat(out_file, mat_file);
}
