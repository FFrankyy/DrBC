#ifndef PREPAREBATCHGRAPH_H_
#define PREPAREBATCHGRAPH_H_

#include "graph.h"
#include "graph_struct.h"
#include "graphUtil.h"
#include "utils.h"
#include <random>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <set>
#include <math.h>
#include <iterator>
#include <functional>


//#include <chrono>


class sparseMatrix
{
 public:
    sparseMatrix();
    ~sparseMatrix();
    std::vector<int> rowIndex;
    std::vector<int> colIndex;
    std::vector<double> value;
    int rowNum;
    int colNum;
};

class PrepareBatchGraph{
public:
    PrepareBatchGraph(int aggregatorID);
    ~PrepareBatchGraph();
    void SetupBatchGraph(std::vector< std::shared_ptr<Graph> > g_list);

    std::shared_ptr<sparseMatrix> n2nsum_param;
    std::shared_ptr<sparseMatrix> n2esum_param;
    std::shared_ptr<sparseMatrix> subgsum_param;
    std::vector<std::vector<int>> neighbor_param;
    std::vector<  std::vector<double>  > node_feat;
    std::vector< std::vector<double> > aux_feat;
    std::vector<std::vector<int>> idx_map_list;
    std::vector<int> size_list;
    std::vector<std::pair<int,int>> subgraph_id_span;
    GraphStruct graph;
    int aggregatorID;
};

bool cmp(std::pair<int, int>a, std::pair<int, int>b);

std::shared_ptr<sparseMatrix> n2n_construct(GraphStruct* graph, int aggregatorID);

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph, std::vector<std::pair<int,int>>& subgraph_id_span);

#endif