#include "metrics.h"
#define max(x, y) (x > y ? x : y)
#define min(x, y) (x > y ? y : x)

Metrics::Metrics()
{
}


double Metrics::MeanSquareError(std::vector<double> real_data,std::vector<double> predict_data){
    double sum_error=0.0;
    assert(real_data.size()==predict_data.size());
    int data_length = (int)real_data.size();
    for(int i = 0;i < data_length; ++i){
        double singleError = (real_data[i]-predict_data[i])*(real_data[i]-predict_data[i]);
        sum_error += singleError;
    }
    return sum_error/(double)data_length;
}

double Metrics::AvgError(std::vector<double> real_data,std::vector<double> predict_data){
    double sum_error=0.0;
    assert(real_data.size()==predict_data.size());
    int data_length = (int)real_data.size();
    for(int i = 0;i < data_length; ++i){
        double singleError = abs(real_data[i]-predict_data[i]);
        sum_error += singleError;
    }
    return sum_error/(double)data_length;
}

double Metrics::MaxError(std::vector<double> real_data,std::vector<double> predict_data){
    assert(real_data.size()==predict_data.size());
    double max_error = 0.0;
    int data_length = (int)real_data.size();
    for(int i = 0;i < data_length; ++i){
        double singleError = abs(real_data[i]-predict_data[i]);
        if (singleError > max_error)
        {
            max_error = singleError;
        }
    }
    return max_error;
}

double Metrics::RankTopK(std::vector<double> real_data, std::vector<double> predict_data, int K){
    assert(real_data.size()==predict_data.size());
    std::vector<double> real_data_copy =  real_data;
    std::vector<double> predict_data_copy = predict_data;
    int data_length = (int)real_data.size();
    std::vector<double>::iterator it;

    std::set<double>::iterator iter;
    std::set<double> real_topKSet;
    std::set<double> predict_topKSet;

    std::nth_element(real_data_copy.begin(),real_data_copy.begin()+K,real_data_copy.end(),std::greater<double>());
    std::nth_element(predict_data_copy.begin(),predict_data_copy.begin()+K,predict_data_copy.end(),std::greater<double>());


    for (it=real_data_copy.begin(); it<real_data_copy.begin()+K; ++it){
        real_topKSet.insert(*it);
    }


    for (it=predict_data_copy.begin(); it<predict_data_copy.begin()+K; ++it){
        predict_topKSet.insert(*it);
    }

    int hit_num = 0;

    for(int i =0;i<data_length;++i){
        if((real_topKSet.count(real_data[i])!=0)&&(predict_topKSet.count(predict_data[i])!=0))
        {
            hit_num += 1;
        }
    }
    return min((double)hit_num/(double)K, 1.0);
}
