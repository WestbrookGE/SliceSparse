#pragma once

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>

#include "config.h"
#include "utils.h"

using namespace std;

/* ------------------- Sparse and dense matrix and vector resources ---------------------- */

template <typename T>
class SVec // a sparse column-vector of type T
{
public:
	_int nr;
	_int size;
	pair<_int,T>* data;

	SVec()
	{
		nr = 0;
		size = 0;
		data = NULL;
	}

	SVec(_int nr, _int size)
	{
		this->nr = nr;
		this->size = size;
		data = new pair<_int,T>[size];
	}

	~SVec()
	{
		delete [] data;
	}

	pair<_int,T>& operator[](const _int i)
	{
		return data[i];
	}

};

template <typename T>
class SMat // a column-major sparse matrix of type T
{
public:
	_int nc;
	_int nr;
	_int* size;
	pair<_int,T>** data;

	SMat()
	{
		nc = 0;
		nr = 0;
		size = NULL;
		data = NULL;
	}

	SMat(_int nr, _int nc)
	{
		this->nr = nr;
		this->nc = nc;
		size = new _int[nc]();
		data = new pair<_int,T>*[nc];
		for(_int i=0; i<nc; i++)
			data[i] = NULL;
	}
	
	SMat(SMat<T>* mat)
	{
		nc = mat->nc;
		nr = mat->nr;
		size = new _int[nc];

		for(_int i=0; i<nc; i++)
			size[i] = mat->size[i];

		data = new pair<_int,T>*[nc];
		for(_int i=0; i<nc; i++)
		{
			data[i] = new pair<_int,T>[size[i]];
			for(_int j=0; j<size[i]; j++)
			{
				data[i][j] = mat->data[i][j];
			}	
		}	
	}
	
	SMat(string fname)
	{
		// Input is in text format
		check_valid_filename(fname,true);

		ifstream fin;
		fin.open(fname);		

		vector<_int> inds;
		vector<T> vals;

		fin>>nc>>nr;
		size = new _int[nc];
		data = new pair<_int,T>*[nc];

		fin.ignore();
		for(_int i=0; i<nc; i++)
		{
			inds.clear();
			vals.clear();
			string line;
			getline(fin,line);
			line += "\n";
			_int pos = 0;
			_int next_pos;

			while(next_pos=line.find_first_of(": \n",pos))
			{
				if((size_t)next_pos==string::npos)
					break;
				inds.push_back(stoi(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

				next_pos = line.find_first_of(": \n",pos);
				if((size_t)next_pos==string::npos)
					break;

				vals.push_back(stof(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

			}

			assert(inds.size()==vals.size());
			assert(inds.size()==0 || inds[inds.size()-1]<nr);

			size[i] = inds.size();
			data[i] = new pair<_int,T>[inds.size()];

			for(_int j=0; j<size[i]; j++)
			{
				data[i][j].first = inds[j];
				data[i][j].second = (T)vals[j];
			}
		}	

		fin.close();
	}

	SMat(string fname, _bool binary=false)
	{
		// Input is in binary format
		if (binary)
		{
			check_valid_filename(fname, true);

			std::ifstream fin;
			fin.open(fname, std::ios::binary);

			readBin(fin);

			fin.close();
			return;
		}

		// Input is in text format
		check_valid_filename(fname,true);

		ifstream fin;
		fin.open(fname);		

		vector<_int> inds;
		vector<T> vals;

		fin>>nc>>nr;
		size = new _int[nc];
		data = new pair<_int,T>*[nc];

		fin.ignore();
		for(_int i=0; i<nc; i++)
		{
			inds.clear();
			vals.clear();
			string line;
			getline(fin,line);
			line += "\n";
			_int pos = 0;
			_int next_pos;

			while(next_pos=line.find_first_of(": \n",pos))
			{
				if((size_t)next_pos==string::npos)
					break;
				inds.push_back(stoi(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

				next_pos = line.find_first_of(": \n",pos);
				if((size_t)next_pos==string::npos)
					break;

				vals.push_back(stof(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

			}
			assert(inds.size()==vals.size());
			
			assert(inds.size()==0 || inds[inds.size()-1]<nr);

			size[i] = inds.size();
			data[i] = new pair<_int,T>[inds.size()];

			for(_int j=0; j<size[i]; j++)
			{
				data[i][j].first = inds[j];
				data[i][j].second = (T)vals[j];
			}
		}	

		fin.close();
	}
	SMat(string fname, std::unordered_map<_int,_int> &inst_map, _bool binary=false)
	{
		// Input is in binary format
		if (binary)
		{
			check_valid_filename(fname, true);

			std::ifstream fin;
			fin.open(fname, std::ios::binary);

			readBin(fin, inst_map);

			fin.close();
			return;
		}

		// Input is in text format
		check_valid_filename(fname,true);

		ifstream fin;
		fin.open(fname);		

		vector<_int> inds;
		vector<T> vals;

		_int nc_full;
		fin>>nc_full>>nr;
		nc = inst_map.size();
		size = new _int[nc];
		data = new pair<_int,T>*[nc];

		fin.ignore();
		_int inst;
		for(_int i=0; i<nc_full; i++)
		{
			inds.clear();
			vals.clear();
			string line;
			getline(fin,line);
			line += "\n";
			_int pos = 0;
			_int next_pos;
			if(inst_map.count(i)==0)
				continue;
			
			while(next_pos=line.find_first_of(": \n",pos))
			{
				if((size_t)next_pos==string::npos)
					break;
				inds.push_back(stoi(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

				next_pos = line.find_first_of(": \n",pos);
				if((size_t)next_pos==string::npos)
					break;

				vals.push_back(stof(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

			}

			assert(inds.size()==vals.size());
			assert(inds.size()==0 || inds[inds.size()-1]<nr);
			inst = inst_map[i];
			
			size[inst] = inds.size();
			data[inst] = new pair<_int,T>[inds.size()];

			for(_int j=0; j<size[inst]; j++)
			{
				data[inst][j].first = inds[j];
				data[inst][j].second = (T)vals[j];
			}
		}	

		fin.close();
	}

	void readBin(std::ifstream& fin)
	{
		fin.read((char *)(&(nc)), sizeof(_int));
		fin.read((char *)(&(nr)), sizeof(_int));

		size = new _int[nc]();
		data = new std::pair<_int, T>*[nc]();
		
		for (_int column = 0; column < (nc); ++column) 
		{
			fin.read((char *)(&(size[column])), sizeof(_int));
			data[column] = new std::pair<_int, T>[size[column]]();
			for (_int row = 0; row < (size[column]); ++row) 
			{
				fin.read((char *)(&(data[column][row].first)), sizeof(_int));
				fin.read((char *)(&(data[column][row].second)), sizeof(T));
			}
		}
	}
	
	void readBin(std::ifstream& fin, std::unordered_map<_int,_int> &inst_map)
	{
		_int nc_full;
		fin.read((char *)(&(nc_full)), sizeof(_int));
		fin.read((char *)(&(nr)), sizeof(_int));
		
		nc = inst_map.size();

		size = new _int[nc]();
		data = new std::pair<_int, T>*[nc]();
		_int col_size, column;
		long long int siz;
		for (_int i = 0; i < (nc_full); ++i) 
		{
			
			fin.read((char *)(&(col_size)), sizeof(_int));
			if(inst_map.count(i)==0)
			{
				siz = col_size*(sizeof(_int)+sizeof(T));
				fin.ignore(siz);
			}
			else
			{
				column = inst_map[i];
				size[column] = col_size;
				data[column] = new std::pair<_int, T>[size[column]]();
				for (_int row = 0; row < (size[column]); ++row) 
				{
					fin.read((char *)(&(data[column][row].first)), sizeof(_int));
					fin.read((char *)(&(data[column][row].second)), sizeof(T));
				}
			}
		}
	}


	_float get_ram()
	{
		_float ram = 0;
		ram += 4*2 + 8*2 + 4*nc + 8*nc;

		for( _int i=0; i<nc; i++ )
			ram += 8*size[i];

		return ram;
	}
	
	void append_mat_columnwise(SMat<T>* tmat)
	{
		assert(nr==tmat->nr);
		int new_nc = nc + tmat->nc;
		_int* temp_size = size;
		pair<_int,T>** temp_data = data;

		size = new _int[new_nc]();
		data = new pair<_int,T>*[new_nc];
		for(_int i=0;i<nc;i++)
		{
			size[i] = temp_size[i];
			data[i] = temp_data[i];
		}

		for(_int i=0;i<tmat->nc;i++)
		{
			size[nc+i] = tmat->size[i];
			data[nc+i] = tmat->data[i];
		}
		nc += tmat->nc;
	}	

	SMat<T>* transpose()
	{
		SMat<T>* tmat = new SMat<T>;
		tmat->nr = nc;
		tmat->nc = nr;
		tmat->size = new _int[tmat->nc]();
		tmat->data = new pair<_int,T>*[tmat->nc];

		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				tmat->size[data[i][j].first]++;
			}
		}

		for(_int i=0; i<tmat->nc; i++)
		{
			tmat->data[i] = new pair<_int,T>[tmat->size[i]];
		}

		_int* count = new _int[tmat->nc]();
		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				_int ind = data[i][j].first;
				T val = data[i][j].second;

				tmat->data[ind][count[ind]].first = i;
				tmat->data[ind][count[ind]].second = val;
				count[ind]++;
			}
		}

		delete [] count;
		return tmat;
	}
	
	vector<_int> remove_columns_with_no_data()
	{
		_int ctr = 0;
		vector<_int> removed_cols;
		for(_int i=0; i<nc; i++)
		{
			if(size[i]==0)
			{
				removed_cols.push_back(i);
				continue;
			}
			Realloc(size[ctr],size[i],data[ctr]);
			size[ctr] = size[i];
			for(_int j=0;j<size[i];j++)
			{
				data[ctr][j].first = data[i][j].first;
				data[ctr][j].second = data[i][j].second;
			}
			
			ctr++;
		}
		for(_int i=ctr;i<nc;i++)
			delete [] data[i];			
		nc = ctr;
		return removed_cols;
	}
	
	void threshold( _float th )
	{
		for( _int i=0; i<nc; i++ )
		{
			_int count = 0;
			for( _int j=0; j<size[i]; j++ )
				count += fabs( data[i][j].second )>th;

			pair<_int,T>* newvec = new pair<_int,T>[count];
			count = 0;
			for( _int j=0; j<size[i]; j++ )
			{
				_int id = data[i][j].first;
				T val = data[i][j].second;
				if( fabs(val)>th )
					newvec[ count++ ] = make_pair( id, val );
			}
			size[i] = count;
			delete [] data[i];
			data[i] = newvec;
		}
	}
	void unit_normalize_columns()
	{
		for(_int i=0; i<nc; i++)
		{
			T normsq = 0;
			for(_int j=0; j<size[i]; j++)
				normsq += SQ(data[i][j].second);
			normsq = sqrt(normsq);

			if(normsq==0)
				normsq = 1;

			for(_int j=0; j<size[i]; j++)
				data[i][j].second /= normsq;
		}
	}

	vector<T> column_norms()
	{
		vector<T> norms(nc,0);

		for(_int i=0; i<nc; i++)
		{
			T normsq = 0;
			for(_int j=0; j<size[i]; j++)
				normsq += SQ(data[i][j].second);
			norms[i] = sqrt(normsq);
		}

		return norms;
	}

	~SMat()
	{
		delete [] size;
		for(_int i=0; i<nc; i++)
			delete [] data[i];
		delete [] data;
	}

	void write(string fname)
	{
		check_valid_filename(fname,false);

		ofstream fout;
		fout.open(fname);

		fout<<nc<<" "<<nr<<endl;

		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				if(j==0)
					fout<<data[i][j].first<<":"<<data[i][j].second;
				else
					fout<<" "<<data[i][j].first<<":"<<data[i][j].second;
			}
			fout<<endl;
		}

		fout.close();
	}
	
	void writeBin(std::ofstream& fout)
	{
		fout.write((char *)(&(nc)), sizeof(_int));
		fout.write((char *)(&(nr)), sizeof(_int));

		for (_int column = 0; column < (nc); ++column) {
			fout.write((char *)(&(size[column])), sizeof(_int));
			for (_int row = 0; row < (size[column]); ++row) 
			{
				fout.write((char *)(&(data[column][row].first)), sizeof(_int));
				fout.write((char *)(&(data[column][row].second)), sizeof(T));
			}
		}
	}
	
	void write(std::string fname, _bool binary)
	{
		if (binary)
		{
			check_valid_filename(fname, false);

			std::ofstream fout;
			fout.open(fname, std::ios::binary);

			writeBin(fout);

			fout.close();
		}
		else
		{
			write(fname);
		}
	}
	
	void add(SMat<T>* smat)
	{
		if(nc != smat->nc || nr != smat->nr)
		{
			cerr<<"SMat::add : Matrix dimensions do not match"<<endl;
			cerr<<"Matrix 1: "<<nc<<" x "<<nr<<endl;
			cerr<<"Matrix 2: "<<smat->nc<<" x "<<smat->nr<<endl;
			exit(1);
		}

		bool* ind_mask = new bool[nr]();
		T* sum = new T[nr]();

		for(_int i=0; i<nc; i++)
		{
			vector<_int> inds;
			for(_int j=0; j<size[i]; j++)
			{
				_int ind = data[i][j].first;
				T val = data[i][j].second;

				sum[ind] += val;
				if(!ind_mask[ind])
				{
					ind_mask[ind] = true;
					inds.push_back(ind);
				}
			}

			for(_int j=0; j<smat->size[i]; j++)
			{
				_int ind = smat->data[i][j].first;
				T val = smat->data[i][j].second;

				sum[ind] += val;
				if(!ind_mask[ind])
				{
					ind_mask[ind] = true;
					inds.push_back(ind);
				}
			}

			sort(inds.begin(), inds.end());
			Realloc(size[i], inds.size(), data[i]);
			for(_int j=0; j<inds.size(); j++)
			{
				_int ind = inds[j];
				data[i][j] = make_pair(ind,sum[ind]);
				ind_mask[ind] = false;
				sum[ind] = 0;
			}
			size[i] = inds.size();
		}

		delete [] ind_mask;
		delete [] sum;
	}

	SMat<T>* prod(SMat<T>* mat2)
	{
		_int dim1 = nr;
		_int dim2 = mat2->nc;

		assert(nc==mat2->nr);

		SMat<T>* prodmat = new SMat<T>(dim1,dim2);
		vector<T> sum(dim1,0);

		for(_int i=0; i<dim2; i++)
		{
			vector<_int> indices;
			for(_int j=0; j<mat2->size[i]; j++)
			{
				_int ind = mat2->data[i][j].first;
				T prodval = mat2->data[i][j].second;

				for(_int k=0; k<size[ind]; k++)
				{
					_int id = data[ind][k].first;
					T val = data[ind][k].second;

					if(sum[id]==0)
						indices.push_back(id);

					sum[id] += val*prodval;
				}
			}

			sort(indices.begin(), indices.end());

			_int siz = indices.size();
			prodmat->size[i] = siz;
			prodmat->data[i] = new pair<_int,T>[siz];

			for(_int j=0; j<indices.size(); j++)
			{
				_int id = indices[j];
				T val = sum[id];
				prodmat->data[i][j] = make_pair(id,val);
				sum[id] = 0;
			}
		}

		return prodmat;
	}
};


template <typename T>
class DMat // a column-major dense matrix of type T
{
public:
	_int nc;
	_int nr;
	float** data;

	DMat()
	{
		nc = 0;
		nr = 0;
		data = NULL;
	}

	DMat(_int nr, _int nc)
	{
		this->nc = nc;
		this->nr = nr;
		data = new T*[nc];
		for(_int i=0; i<nc; i++)
			data[i] = new T[nr]();
	}

	DMat(SMat<T>* mat)
	{
		nc = mat->nc;
		nr = mat->nr;
		data = new T*[nc];
		for(_int i=0; i<nc; i++)
			data[i] = new T[nr]();

		for(_int i=0; i<mat->nc; i++)
		{
			pair<_int,T>* vec = mat->data[i];
			for(_int j=0; j<mat->size[i]; j++)
			{
				data[i][vec[j].first] = vec[j].second;
			}
		}
	}
	void unit_normalize_columns()
	{
		for(_int i=0; i<nc; i++)
		{
			T normsq = 0;
			for(_int j=0; j<nr; j++)
				normsq += SQ(data[i][j]);
			normsq = sqrt(normsq);

			if(normsq==0)
				normsq = 1;

			for(_int j=0; j<nr; j++)
				data[i][j] /= normsq;
		}
	}
	DMat<T>* prod(SMat<T>* mat2)
	{
		_int dim1 = nr;
		_int dim2 = mat2->nc;

		assert(nc==mat2->nr);
		DMat<T>* prodmat = new DMat<T>(dim1,dim2);
		vector<T> sum(dim1,0);

		for(_int i=0; i<dim2; i++)
		{
			for(_int j=0; j<mat2->size[i]; j++)
			{
				_int ind = mat2->data[i][j].first;
				T prodval = mat2->data[i][j].second;

				for(_int k=0; k<nr; k++)
				{
					T val = data[ind][k];
					sum[k] += val*prodval;
				}
			}

			for(_int j=0; j<dim1; j++)
			{
				prodmat->data[i][j] = sum[j];
				sum[j] = 0;
			}
		}

		return prodmat;
	}
	DMat(string fname, _bool binary=false)
	{
		// Input is in binary format
		if (binary)
		{
			check_valid_filename(fname, true);

			std::ifstream fin;
			fin.open(fname, std::ios::binary);

			readBin(fin);

			fin.close();
			return;
		}

		// Input is in text format
		check_valid_filename(fname,true);

		ifstream fin;
		fin.open(fname);		

		vector<T> vals;

		fin>>nc>>nr;
		data = new T*[nc];

		fin.ignore();
		for(_int i=0; i<nc; i++)
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

			data[i] = new T[nr];

			for(_int j=0; j<nr; j++)
				data[i][j] = (T)vals[j];
		}	

		fin.close();
	}
	DMat(string fname, std::unordered_map<_int,_int> &inst_map, _bool binary=false)
	{
		_int nc_full;
		// Input is in binary format
		if (binary)
		{
			check_valid_filename(fname, true);

			std::ifstream fin;
			fin.open(fname, std::ios::binary);

			readBin(fin, inst_map);

			fin.close();
			return;
		}

		// Input is in text format
		check_valid_filename(fname,true);

		ifstream fin;
		fin.open(fname);		

		vector<T> vals;

		fin>>nc_full>>nr;
		nc = inst_map.size();
		data = new T*[nc];

		fin.ignore();
		_int inst;
		for(_int i=0; i<nc_full; i++)
		{
			vals.clear();
			string line;
			getline(fin,line);
			line += "\n";
			_int pos = 0;
			_int next_pos;
			if(inst_map.count(i)==0)
				continue;
			
			while(next_pos=line.find_first_of(" \n",pos))
			{
				if((size_t)next_pos==string::npos)
					break;

				vals.push_back(stof(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;

			}

			assert(vals.size()==nr);
			inst = inst_map[i];
			
			data[inst] = new T[nr];

			for(_int j=0; j<nr; j++)
				data[inst][j] = (T)vals[j];
		}	

		fin.close();
	}

	void readBin(std::ifstream& fin)
	{
		fin.read((char *)(&(nc)), sizeof(_int));
		fin.read((char *)(&(nr)), sizeof(_int));
		
		data = new T*[nc];

		for (_int column = 0; column < (nc); ++column) 
		{
			data[column] = new T[nr]();
			for (_int row = 0; row < nr; ++row) 
			{
				fin.read((char *)(&(data[column][row])), sizeof(T));
			}
		}
	}
	
	void readBin(std::ifstream& fin, std::unordered_map<_int,_int> &inst_map)
	{
		_int nc_full;
		fin.read((char *)(&(nc_full)), sizeof(_int));
		fin.read((char *)(&(nr)), sizeof(_int));
		
		nc = inst_map.size();

		data = new T*[nc];
		_int col_size, column;
		long long int siz;
		for (_int i = 0; i < (nc_full); ++i) 
		{
			if(inst_map.count(i)==0)
			{
				siz = nr*(sizeof(T));
				fin.ignore(siz);
			}
			else
			{
				column = inst_map[i];
				data[column] = new T[nr]();
				for (_int row = 0; row < (nr); ++row) 
				{
					fin.read((char *)(&(data[column][row])), sizeof(T));
				}
			}
		}
	}
	void write(string fname)
	{
		check_valid_filename(fname,false);

		ofstream fout;
		fout.open(fname);

		fout<<nc<<" "<<nr<<endl;

		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<nr; j++)
			{
				if(j==0)
					fout<<data[i][j];
				else
					fout<<" "<<data[i][j];
			}
			fout<<endl;
		}

		fout.close();
	}
	
	void writeBin(std::ofstream& fout)
	{
		fout.write((char *)(&(nc)), sizeof(_int));
		fout.write((char *)(&(nr)), sizeof(_int));

		for (_int column = 0; column < (nc); ++column) 
		{
			for (_int row = 0; row < (nr); ++row) 
				fout.write((char *)(&(data[column][row])), sizeof(T));
		}
	}
	void write_in_numpy_format(string fname)
	{
		check_valid_filename(fname, false);
		std::ofstream fout;
		fout.open(fname, std::ios::binary);
		for (_int column = 0; column < (nc); ++column) 
		{
			for (_int row = 0; row < (nr); ++row) 
				fout.write((char *)(&(data[column][row])), sizeof(T));
		}
		fout.close();
	}
	
	void write(std::string fname, _bool binary)
	{
		if (binary)
		{
			check_valid_filename(fname, false);

			std::ofstream fout;
			fout.open(fname, std::ios::binary);

			writeBin(fout);

			fout.close();
		}
		else
		{
			write(fname);
		}
	}


	~DMat()
	{
		for(_int i=0; i<nc; i++)
			delete [] data[i];
		delete [] data;
	}
};

class IMat // a column-major sparse matrix with indices only
{
public:
	_int nc;
	_int nr;
	_int* size;
	_int** data;

	IMat()
	{
		nc = 0;
		nr = 0;
		size = NULL;
		data = NULL;
	}

	IMat(_int nr, _int nc)
	{
		this->nr = nr;
		this->nc = nc;
		size = new _int[nc]();
		data = new _int*[nc];
		for(_int i=0; i<nc; i++)
			data[i] = NULL;
	}
	
	IMat(string fname)
	{
		check_valid_filename(fname,true);

		ifstream fin;
		fin.open(fname);		

		vector<_int> inds;
		fin>>nc>>nr;
		size = new _int[nc];
		data = new _int*[nc];

		fin.ignore();
		for(_int i=0; i<nc; i++)
		{
			inds.clear();
			string line;
			getline(fin,line);
			line += "\n";
			_int pos = 0;
			_int next_pos;

			while(next_pos=line.find_first_of(" \n",pos))
			{
				if((size_t)next_pos==string::npos)
					break;
				inds.push_back(stoi(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;
			}

			assert(inds.size()==0 || inds[inds.size()-1]<nr);

			size[i] = inds.size();
			data[i] = new _int[inds.size()];

			for(_int j=0; j<size[i]; j++)
			{
				data[i][j] = inds[j];
			}
		}	

		fin.close();
	}
	IMat(string fname, _bool binary=false)
	{
		// Input is in binary format
		if (binary)
		{
			check_valid_filename(fname, true);

			std::ifstream fin;
			fin.open(fname, std::ios::binary);

			readBin(fin);

			fin.close();
			return;
		}

		// Input is in text format
		check_valid_filename(fname,true);

		ifstream fin;
		fin.open(fname);		

		vector<_int> inds;
		fin>>nc>>nr;
		size = new _int[nc];
		data = new _int*[nc];

		fin.ignore();
		for(_int i=0; i<nc; i++)
		{
			inds.clear();
			string line;
			getline(fin,line);
			line += "\n";
			_int pos = 0;
			_int next_pos;

			while(next_pos=line.find_first_of(" \n",pos))
			{
				if((size_t)next_pos==string::npos)
					break;
				inds.push_back(stoi(line.substr(pos,next_pos-pos)));
				pos = next_pos+1;
			}

			assert(inds.size()==0 || inds[inds.size()-1]<nr);

			size[i] = inds.size();
			data[i] = new _int[inds.size()];

			for(_int j=0; j<size[i]; j++)
			{
				data[i][j] = inds[j];
			}
		}	

		fin.close();
	}

	void readBin(std::ifstream& fin)
	{
		fin.read((char *)(&(nc)), sizeof(_int));
		fin.read((char *)(&(nr)), sizeof(_int));

		size = new _int[nc]();
		data = new _int*[nc];
		for (_int column = 0; column < (nc); ++column) {
			fin.read((char *)(&(size[column])), sizeof(_int));
			data[column] = new _int[size[column]]();
			for (_int row = 0; row < (size[column]); ++row) 
			{
				fin.read((char *)(&(data[column][row])), sizeof(_int));
			}
		}
	}
	void write(string fname)
	{
		check_valid_filename(fname,false);

		ofstream fout;
		fout.open(fname);

		fout<<nc<<" "<<nr<<endl;

		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				if(j==0)
					fout<<data[i][j];
				else
					fout<<" "<<data[i][j];
			}
			fout<<endl;
		}

		fout.close();
	}
	
	void writeBin(std::ofstream& fout)
	{
		fout.write((char *)(&(nc)), sizeof(_int));
		fout.write((char *)(&(nr)), sizeof(_int));

		for (_int column = 0; column < (nc); ++column) {
			fout.write((char *)(&(size[column])), sizeof(_int));
			for (_int row = 0; row < (size[column]); ++row) 
			{
				fout.write((char *)(&(data[column][row])), sizeof(_int));
			}
		}
	}
	
	void write(std::string fname, _bool binary)
	{
		if (binary)
		{
			check_valid_filename(fname, false);

			std::ofstream fout;
			fout.open(fname, std::ios::binary);

			writeBin(fout);

			fout.close();
		}
		else
		{
			write(fname);
		}
	}

	void append_mat_columnwise(IMat* tmat)
	{
		assert(nr==tmat->nr);
		int new_nc = nc + tmat->nc;
		_int* temp_size = size;
		_int** temp_data = data;

		size = new _int[new_nc]();
		data = new _int*[new_nc];
		for(_int i=0;i<nc;i++)
		{
			size[i] = temp_size[i];
			data[i] = temp_data[i];
		}

		for(_int i=0;i<tmat->nc;i++)
		{
			size[nc+i] = tmat->size[i];
			data[nc+i] = tmat->data[i];
		}
		nc += tmat->nc;
	}	
	IMat* transpose()
	{
		IMat* tmat = new IMat;
		tmat->nr = nc;
		tmat->nc = nr;
		tmat->size = new _int[tmat->nc]();
		tmat->data = new _int*[tmat->nc];

		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				tmat->size[data[i][j]]++;
			}
		}

		for(_int i=0; i<tmat->nc; i++)
		{
			tmat->data[i] = new _int[tmat->size[i]];
		}

		_int* count = new _int[tmat->nc]();
		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				_int ind = data[i][j];
				tmat->data[ind][count[ind]] = i;
				count[ind]++;
			}
		}

		delete [] count;
		return tmat;
	}
	SMat<_float>* toSMat()
	{
		SMat<_float>* smat = new SMat<_float>;
		smat->nr = nr;
		smat->nc = nc;
		smat->size = new _int[smat->nc]();
		smat->data = new pair<_int,_float>*[smat->nc];

		for(_int i=0; i<smat->nc; i++)
		{
			smat->size[i] = size[i];
			smat->data[i] = new pair<_int,_float>[smat->size[i]];
		}

		for(_int i=0; i<nc; i++)
		{
			for(_int j=0; j<size[i]; j++)
			{
				smat->data[i][j].first = data[i][j];
				smat->data[i][j].second = (_float)1;
			}
		}
		return smat;
	}

	~IMat()
	{
		for(_int i=0; i<nc; i++)
			delete [] data[i];
		delete [] data;
	}
};
typedef vector<_int> VecI;
typedef vector<_float> VecF;
typedef vector<_double> VecD;
typedef vector<pairII> VecII;
typedef vector<pairIF> VecIF;
typedef vector<_bool> VecB;
typedef SMat<_float> SMatF;
typedef SMat<_int> SMatI;
typedef DMat<_float> DMatF;
