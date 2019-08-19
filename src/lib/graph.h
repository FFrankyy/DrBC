#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>
class Graph
{
public:
    Graph();
    Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to);
    ~Graph();
    int num_nodes;
    int num_edges;
    std::vector< std::vector< int > > adj_list;
    std::vector< std::pair<int, int> > edge_list;
    std::set< std::pair<int,int> > edge_set;
};

class GSet
{
public:
    GSet();
    ~GSet();
    void InsertGraph(int gid, std::shared_ptr<Graph> graph);
    int GetSampleID();
    std::shared_ptr<Graph> Get(int gid);
    void Clear();
    std::map<int, std::shared_ptr<Graph> > graph_pool;
};

extern GSet GSetTrain;
extern GSet GSetTest;

#endif