import numpy
import sys
import nmslib
import time
import math
import pdb
from xclib.data import data_utils
import hnswlib

lbl_ft_file = sys.argv[1]
model_file = sys.argv[2]
M = int(sys.argv[3])
efC = int(sys.argv[4])
num_threads = int(sys.argv[5])
num_ft = int(sys.argv[6])
metric_space = sys.argv[7]

start = time.time()
data = data_utils.read_sparse_file(lbl_ft_file)
end = time.time()
start = time.time()
index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
index.addDataPointBatch(data)
index.createIndex({'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC})
end = time.time()
print('Training time of ANNS datastructure = %f'%(end-start))
nmslib.saveIndex(index,model_file)
