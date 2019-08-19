from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from libc.stdlib cimport malloc
from scipy import stats
from cpython cimport bool
from libc cimport math
cimport cython
import numpy as np
cimport numpy as np
import scipy as sp
import scipy.stats
from numpy cimport ndarray, int64_t, float64_t, intp_t


cdef class py_Metrics:
    cdef shared_ptr[Metrics] inner_Metrics

    def __cinit__(self):
        self.inner_Metrics = shared_ptr[Metrics](new Metrics())

    def MeanSquareError(self,real_data,predict_data):
        return deref(self.inner_Metrics).MeanSquareError(real_data, predict_data)

    def AvgError(self,real_data,predict_data):
        diff = np.array(real_data) - np.array(predict_data)
        return np.mean(abs(diff))

    def MaxError(self,real_data,predict_data):
        diff = np.array(real_data) - np.array(predict_data)
        return np.max(abs(diff))

    def RankTopK(self, real_data, predict_data, K):
        if K < 1:
            TopK = max(int(len(real_data) * K),1)
        else:
            TopK = K
        TopK = int(TopK)
        return deref(self.inner_Metrics).RankTopK(real_data, predict_data, TopK)

    def RankKendal(self, x, y, initial_lexsort=None, nan_policy='propagate'):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        if x.size != y.size:
            raise ValueError("All inputs to `kendalltau` must be of the same size, "
                             "found x-size %s and y-size %s" % (x.size, y.size))
        elif not x.size or not y.size:
            return np.nan  # Return NaN if arrays are empty

        def count_rank_tie(ranks):
            cnt = np.bincount(ranks).astype('int64', copy=False)
            cnt = cnt[cnt > 1]
            return ((cnt * (cnt - 1) // 2).sum(),
                    (cnt * (cnt - 1.) * (cnt - 2)).sum(),
                    (cnt * (cnt - 1.) * (2 * cnt + 5)).sum())

        size = x.size
        perm = np.argsort(y)  # sort on y and convert y to dense ranks
        x, y = x[perm], y[perm]
        y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

        # stable sort on x and convert x to dense ranks
        perm = np.argsort(x, kind='mergesort')
        x, y = x[perm], y[perm]
        x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

        dis = self.kendall_dis(x, y)  # discordant pairs

        obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
        cnt = np.diff(np.where(obs)[0]).astype('int64', copy=False)

        ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
        xtie, x0, x1 = count_rank_tie(x)  # ties in x, stats
        ytie, y0, y1 = count_rank_tie(y)  # ties in y, stats

        tot = (size * (size - 1)) // 2

        con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
        tau = con_minus_dis / tot
        # Limit range to fix computational errors
        tau = min(1., max(-1., tau))

        return min(1., max(-1., tau))

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def kendall_dis(self, intp_t[:] x, intp_t[:] y):
        cdef:
            intp_t sup = 1 + np.max(y)
            intp_t[::1] arr = np.zeros(sup, dtype=np.intp)
            intp_t i = 0, k = 0, size = x.size, idx
            int64_t dis = 0

        with nogil:
            while i < size:
                while k < size and x[i] == x[k]:
                    dis += i
                    idx = y[k]
                    while idx != 0:
                        dis -= arr[idx]
                        idx = idx & (idx - 1)

                    k += 1

                while i < k:
                    idx = y[i]
                    while idx < sup:
                        arr[idx] += 1
                        idx += idx & -idx
                    i += 1
        return dis