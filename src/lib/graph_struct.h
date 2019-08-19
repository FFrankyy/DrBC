#ifndef GRAPH_STRUCT_H
#define GRAPH_STRUCT_H

#include <vector>
#include <map>
#include <iostream>
#include <cassert>
#include <algorithm>

template<typename T>
class LinkedTable
{
public:

		LinkedTable();
        ~LinkedTable();
		void AddEntry(int head_id, T content);
		void Resize(int new_n);
		int n;
		std::vector< std::vector<T> > head;
private:
		int ncap;
};


class GraphStruct
{
public:
	GraphStruct();
	~GraphStruct();

	void AddEdge(int idx, int x, int y);

	void AddNode(int subg_id, int n_idx);

	void Resize(unsigned _num_subgraph, unsigned _num_nodes = 0);

	LinkedTable< std::pair<int, int> > *out_edges;

	LinkedTable< std::pair<int, int> > *in_edges;

	LinkedTable< int >* subgraph;

	std::vector< std::pair<int, int> > edge_list;

	unsigned num_nodes;

	unsigned num_edges;

	unsigned num_subgraph;
};

extern GraphStruct batch_graph;

#endif