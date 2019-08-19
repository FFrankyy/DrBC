from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from graph cimport Graph
import gc
from libc.stdlib cimport free

cdef class py_Utils:
    cdef shared_ptr[Utils] inner_Utils
    def __cinit__(self):
        self.inner_Utils = shared_ptr[Utils](new Utils())

    def Betweenness_Batch(self,_g_list):
        cdef shared_ptr[Graph] inner_Graph
        cdef vector[shared_ptr[Graph]] inner_glist
        for _g in _g_list:
            inner_Graph = shared_ptr[Graph](new Graph())
            deref(inner_Graph).num_nodes = _g.num_nodes
            deref(inner_Graph).num_edges = _g.num_edges
            deref(inner_Graph).edge_list = _g.edge_list
            deref(inner_Graph).adj_list = _g.adj_list
            inner_glist.push_back(inner_Graph)
        return deref(self.inner_Utils).Betweenness_Batch(inner_glist)

    def Betweenness(self,_g):
        cdef shared_ptr[Graph] inner_Graph
        inner_Graph =shared_ptr[Graph](new Graph())
        deref(inner_Graph).num_nodes = _g.num_nodes
        deref(inner_Graph).num_edges = _g.num_edges
        deref(inner_Graph).edge_list = _g.edge_list
        deref(inner_Graph).adj_list = _g.adj_list
        return deref(self.inner_Utils).Betweenness(inner_Graph)


    @property
    def bc_log(self):
        return deref(self.inner_Utils).bc_log
    @property
    def bc_bool(self):
        return deref(self.inner_Utils).bc_bool
    
    def convertToLog(self,CB):
        return deref(self.inner_Utils).convertToLog(CB)

