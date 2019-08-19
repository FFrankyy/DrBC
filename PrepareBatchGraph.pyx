from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from libc.stdlib cimport malloc
from libc.stdlib cimport free
from graph cimport Graph
import tensorflow as tf
from scipy.sparse import coo_matrix


cdef class py_sparseMatrix:
    cdef shared_ptr[sparseMatrix] inner_sparseMatrix
    def __cinit__(self):
        self.inner_sparseMatrix =shared_ptr[sparseMatrix](new sparseMatrix())

    @property
    def rowIndex(self):
        return deref(self.inner_sparseMatrix).rowIndex
    @property
    def colIndex(self):
        return deref(self.inner_sparseMatrix).colIndex
    @property
    def value(self):
        return deref(self.inner_sparseMatrix).value
    @property
    def rowNum(self):
        return deref(self.inner_sparseMatrix).rowNum
    @property
    def colNum(self):
        return deref(self.inner_sparseMatrix).colNum


cdef class py_PrepareBatchGraph:
    cdef shared_ptr[PrepareBatchGraph] inner_PrepareBatchGraph
    cdef sparseMatrix matrix
    def __cinit__(self,aggregatorID):
        self.inner_PrepareBatchGraph =shared_ptr[PrepareBatchGraph](new PrepareBatchGraph(aggregatorID))

    def SetupBatchGraph(self,g_list):
        cdef shared_ptr[Graph] inner_Graph
        cdef vector[shared_ptr[Graph]] inner_glist
        for _g in g_list:
            inner_Graph = shared_ptr[Graph](new Graph())
            deref(inner_Graph).num_nodes = _g.num_nodes
            deref(inner_Graph).num_edges = _g.num_edges
            deref(inner_Graph).edge_list = _g.edge_list
            deref(inner_Graph).adj_list = _g.adj_list
            inner_glist.push_back(inner_Graph)
        deref(self.inner_PrepareBatchGraph).SetupBatchGraph(inner_glist)

    @property
    def n2nsum_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).n2nsum_param)
        return self.ConvertSparseToTensor(matrix)

    @property
    def subgsum_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).subgsum_param)
        return self.ConvertSparseToTensor(matrix)

    @property
    def neighbor_param(self):
        return deref(self.inner_PrepareBatchGraph).neighbor_param

    @property
    def aux_feat(self):
        return deref(self.inner_PrepareBatchGraph).aux_feat

    @property
    def node_feat(self):
        return deref(self.inner_PrepareBatchGraph).node_feat

    @property
    def idx_map_list(self):
        return deref(self.inner_PrepareBatchGraph).idx_map_list

    @property
    def subgraph_id_span(self):
        return deref(self.inner_PrepareBatchGraph).subgraph_id_span

    @property
    def size_list(self):
        return deref(self.inner_PrepareBatchGraph).size_list

    def ahead_one(self, a):
        a = list(a)
        b = a.pop(0)
        a.append(b)
        return np.array(a)

    @property
    def pair_ids_src(self):
        size_list = list(deref(self.inner_PrepareBatchGraph).size_list)
        ids_src = []
        offset = 0
        for size in size_list:
            ids_src += list(np.array(list(range(size))+list(np.random.choice(range(size), size=4*size, replace=True)))+offset)
            offset += size
        return ids_src

    @property
    def pair_ids_tgt(self):
        size_list = list(deref(self.inner_PrepareBatchGraph).size_list)
        ids_tgt = []
        offset = 0
        for size in size_list:
            ids_tgt += list(np.array(list(self.ahead_one(np.arange(size)))+list(np.random.choice(range(size), size=4*size, replace=True)))+offset)
            offset += size
        return ids_tgt

    @property
    def aggregatorID(self):
        return deref(self.inner_PrepareBatchGraph).aggregatorID

    cdef ConvertSparseToTensor(self,sparseMatrix matrix):
        rowIndex= matrix.rowIndex
        colIndex= matrix.colIndex
        data= matrix.value
        rowNum= matrix.rowNum
        colNum= matrix.colNum
        indices = np.mat([rowIndex, colIndex]).transpose()
        return tf.SparseTensorValue(indices, data, (rowNum,colNum))




