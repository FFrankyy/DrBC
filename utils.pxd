
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from graph cimport Graph

cdef extern from "./src/lib/utils.h":
    cdef cppclass Utils:

        Utils()

        vector[double] Betweenness_Batch(vector[shared_ptr[Graph]] graph_list)
        vector[double] Betweenness(shared_ptr[Graph] graph)
        vector[double] convertToLog(vector[double] CB)
        vector[double] bc_log
        vector[int] bc_bool

