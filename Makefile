ToolsDir=./Tools/c++/
MetricsDir=./Tools/metrics/
INC=-I$(ToolsDir)
CXXFLAGS=-std=c++11 -O3 -g -fopenmp
all: clean slice_train slice_predict smat_to_dmat dmat_to_smat precision_k nDCG_k
slice_train:
	$(CXX) -o slice_train $(CXXFLAGS) $(INC) slice_train.cpp slice.cpp  
slice_predict:
	$(CXX) -o slice_predict $(CXXFLAGS) $(INC) slice_predict.cpp slice.cpp  
smat_to_dmat:
	$(CXX) -o $(ToolsDir)smat_to_dmat $(CXXFLAGS) $(INC) $(ToolsDir)smat_to_dmat.cpp
dmat_to_smat:
	$(CXX) -o $(ToolsDir)dmat_to_smat $(CXXFLAGS) $(INC) $(ToolsDir)dmat_to_smat.cpp
precision_k:
	$(CXX) -o $(MetricsDir)precision_k $(CXXFLAGS) $(INC) $(MetricsDir)precision_k.cpp	
nDCG_k:
	$(CXX) -o $(MetricsDir)nDCG_k $(CXXFLAGS) $(INC) $(MetricsDir)nDCG_k.cpp	
clean:
	rm -f slice_train slice_predict $(ToolsDir)smat_to_dmat $(ToolsDir)dmat_to_smat $(MetricsDir)precision_k $(MetricsDir)nDCG_k
