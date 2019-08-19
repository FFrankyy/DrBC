from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.pair cimport pair
cdef extern from "./src/lib/metrics.h":

    cdef cppclass Metrics:

        Metrics() except+

        double MeanSquareError(vector[double] real_data, vector[double] predict_data) except+

        double AvgError(vector[double] real_data, vector[double] predict_data) except+

        double MaxError(vector[double] real_data, vector[double] predict_data) except+

        double RankTopK(vector[double] real_data, vector[double] predict_data,double K) except+

