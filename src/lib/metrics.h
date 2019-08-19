#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <math.h>
#include <algorithm>
#include <iterator>
#include <set>
#include <assert.h>
#include <functional>
#include <cstdio>




class Metrics{
    public:

    Metrics();

    double MeanSquareError(std::vector<double> real_data,std::vector<double> predict_data);

    double AvgError(std::vector<double> real_data,std::vector<double> predict_data);

    double MaxError(std::vector<double> real_data,std::vector<double> predict_data);

    double RankTopK(std::vector<double> real_data,std::vector<double> predict_data,int K);

};

#endif