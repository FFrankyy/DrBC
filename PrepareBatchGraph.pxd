from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.pair cimport pair
from graph cimport Graph
cdef extern from "./src/lib/PrepareBatchGraph.h":
    cdef cppclass sparseMatrix:
        sparseMatrix()except+
        vector[int] rowIndex
        vector[int] colIndex
        vector[double] value
        int rowNum
        int colNum

    cdef cppclass PrepareBatchGraph:
        PrepareBatchGraph(int aggregatorID)except+

        void SetupBatchGraph(vector[shared_ptr[Graph] ] g_list)except+

        shared_ptr[sparseMatrix] n2nsum_param
        shared_ptr[sparseMatrix] subgsum_param
        vector[vector[int]]  neighbor_param
        vector[vector[double]]  aux_feat
        vector[vector[double]]  node_feat
        vector[vector[int]] idx_map_list
        vector[int] size_list
        vector[pair[int,int]] subgraph_id_span

        int aggregatorID