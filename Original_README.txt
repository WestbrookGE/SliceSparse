** Please make sure that you read and agree to the terms of license (License.pdf) and copyright (liblinear_COPYRIGHT) before using this software. **

This is the code for the algorithm proposed in our research paper "Slice: Scalable Linear Extreme Classifiers trained on 100 Million Labels for Related Searches" authored by Himanshu Jain, Venkatesh B., Bhanu Teja Chunduri and Manik Varma and published in The Web Search and Data Mining Conference-2019. The code is authored by Himanshu Jain (himanshu.j689@gmail.com).

About Slice
=============
Extreme multi-label learning aims to annotate each data  point with the most relevant subset of labels from an extremely large label set. Slice is an efficient 1-vs-All based extreme classifier that is specially designed for low-dimensional dense features. Slice achieves close to state-of-the-art accuracies while being significantly faster to train and predict than most other extreme classifiers. Slice can efficiently scale to datasets with as many as 100 million labels and 240 million training points. Please refer to the research paper for more details.

This code is made available as is for non-commercial research purposes. Please make sure that you have read the license agreement in LICENSE.doc/pdf. Please do not install or use Slice unless you agree to the terms of the license.

The code for Slice is written in C++ and should compile on 64 bit Windows/Linux machines using a C++11 enabled compiler. The code also uses the publically available implementation of the HNSW algorithm (https://github.com/nmslib/nmslib) to find the Approximate Nearest Neighbors. Installation and usage instructions are provided below. The default parameters provided in the Usage Section work reasonably on the benchmark datasets in the Extreme Classification Repository (http://manikvarma.org/downloads/XC/XMLRepository.html). 

Please contact Himanshu Jain (himanshu.j689@gmail.com) and Manik Varma (manik@microsoft.com) if you have any questions or feedback.

Experimental Results and Datasets
=================================
Please visit the Extreme Classification Repository (http://manikvarma.org/downloads/XC/XMLRepository.html) to download the benchmark datasets (dense feature versions) and compare Slice's performance to baseline algorithms.

Usage
=====
Linux/Windows makefiles for compiling Slice have been provided with the source code. To compile, run "make" (Linux) or "nmake -f Makefile.win" (Windows) in the topmost folder. Run the following commands from inside Slice folder for training and testing.

Training
--------

C++:
	./slice_train [input feature file name] [input label file name] [output model folder name] -m 100 -c 300 -s 300 -k 300 -o 20 -t 1 -C 1 -f 0.000001 -siter 20 -stype 0 -q 0

where:	
	-m = params.M                       :        HNSW M parameter. default=100
	-c = params.efC                     :        HNSW efConstruction parameter. default=300
	-s = params.efS                     :        HNSW efSearch parameter. default=300
	-k = params.num_nbrs                :        Number of labels to be shortlisted per training point according to the generative model. default=300
	-o = params.num_io_threads          :        Number of threads used to write the retrived ANN points to file. default=20
	-t = params.num_threads             :        Number of threads used to train ANNS datastructure and the discriminative classifiers. default=1
	-C = params.classifier_cost         :        Cost co-efficient for linear classifiers            default=1.0 SVM weight co-efficient. default=1.0
	-f = params.classifier_threshold    :        Threshold value for sparsifying linear classifiers' trained weights to reduce model size. default=1e-6
	-siter = params.classifier_maxiter  :        Maximum iterations of algorithm for training linear classifiers. default=20
	-stype = param.classifier_kind      :        Kind of linear classifier to use. 0=L2R_L2LOSS_SVC, 1=L2R_LR (Refer to Liblinear). default=0
	-q = param.quiet				            :        Quiet option to restrict the output for reporting progress and debugging purposes 0=no quiet, 1=quiet		default=[value saved in trained model]

	Feature file should be in dense matrix text format and label file should be in sparse matrix text format (refer to Miscellaneous section).

Testing
-------

C++:
	./slice_predict [feature file name] [model dir name] [output file name] -b 0 -t 1 -q 0

where:
	-s = params.efS                     :        HNSW efSearch parameter. default=[value saved in trained model]
	-k = params.num_nbrs                :        Number of labels to be shortlisted per training point according to the generative model. default=[value saved in trained model]
	-o = params.num_io_threads          :        Number of threads used to write the retrived ANN points to file. default=[value saved in trained model]
	-b = params.b_gen                   :        Bias parameter for the generative model. default=0
	-t = params.num_threads             :        Number of threads. default=[value saved in trained model]
	-q = param.quiet				            :        Quiet option to restrict the output for reporting progress and debugging purposes 0=no quiet, 1=quiet. default=[value saved in trained model]

	Feature file should be in dense matrix text format (refer to Miscellaneous section).

Performance Evaluation
----------------------

Scripts to calculate Precision@k and nDCG@k are available in C++ (in Tools/metrics folder). Following command can be used to compute these metrics:
	./precision_k [test score matrix file name] [test label file name] [K]
	./nDCG_k [test score matrix file name] [test label file name] [K]

For propensity scored metrics use the Matlab scripts. To compile these scripts, execute "make" in the topmost folder from the Matlab terminal.
All the metrics can be computed by executing the following command from Tools/metrics folder:

	[metrics] = get_all_metrics([test score matrix], [test label matrix], [inverse label propensity vector])

Miscellaneous
-------------

* Slice requires different formats for feature and label input files as the features are stored in dense matrix format while labels are stored in sparse format. The first line of both the files contains the number of rows and columns and the subsequent lines contain one data instance per row. For features each line contains D (dimensionality of feature vector), space separated, float values while each line of the label file contains indices of active labels and the corresponding value (always 1 in this case) starting from 0. Scripts to convert features from sparse format to dense format and vice versa are also provided in the Tools/c++ folder. Following commands can be used for the conversions:

			./smat_to_dmat [sparse feature text file] [dense feature text file]
			./dmat_to_smat [dense feature text file] [sparse feature text file]

* Scripts are provided in the 'Tools' folder for sparse matrix inter conversion between Matlab .mat format and text format.
    To read a sparse text matrix into Matlab:

    	[sparse matrix] = read_text_mat([sparse text matrix file name]); 

    To write a Matlab matrix into sparse text format:

    	write_text_mat([Matlab sparse matrix], [output sparse text matrix file name]);

* To generate inverse label propensity weights, run the following command inside 'Tools/metrics' folder on Matlab terminal:

    	[weights vector] = inv_propensity([training label matrix],A,B); 

    A,B are the parameters of the inverse propensity model. Following values are to be used over the benchmark datasets:

    	Wikipedia-500K:  A=0.5,  B=0.4
    	Amazon-670K:     A=0.6,  B=2.6
    	Other:		       A=0.55, B=1.5

Toy Example
===========

The zip file containing the source code also includes the EURLex-4K dataset with 1024 dimensional XMLCNN features as a toy example.
To run Slice on the EURLex-4K dataset, execute "bash sample_run.sh" (Linux) or "sample_run" (Windows) in the Slice folder.
Read the comments provided in the above scripts for better understanding.

