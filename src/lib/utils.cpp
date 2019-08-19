#include "utils.h"
#include "graph.h"
#include "graphUtil.h"
#include <cassert>
#include <random>
#include <algorithm>
#include <set>
#include "stdio.h"
#include <queue>
#include <stack>


Utils::Utils()
{
bc_bool.clear();
bc_log.clear();
}

std::vector<double> Utils::Betweenness_Batch(std::vector<std::shared_ptr<Graph>> _g_list)
{
    std::vector<double> result;
    for(int i = 0 ;i<_g_list.size();++i){
        auto _g = _g_list[i];
        std::vector<double> result_g = Betweenness(_g);
        for(int j = 0;j<result_g.size();++j){
            result.push_back(result_g[j]);
        }
    }
    return result;
}

std::vector<double> Utils::Betweenness(std::shared_ptr<Graph> _g) {


	int i, j, u, v;
	int Long_max = 4294967295;
	int nvertices = _g->num_nodes;	// The number of vertices in the network
	std::vector<double> CB;
    double norm=(double)(nvertices-1)*(double)(nvertices-2);

    bc_bool.resize(nvertices);
    bc_log.resize(nvertices);
	CB.resize(nvertices);

	std::vector<int> d;								// A vector storing shortest distance estimates
	std::vector<int> sigma;							// sigma is the number of shortest paths
	std::vector<double> delta;							// A vector storing dependency of the source vertex on all other vertices
	std::vector< std::vector <int> > PredList;			// A list of predecessors of all vertices

	std::queue <int> Q;								// A priority queue soring vertices
	std::stack <int> S;								// A stack containing vertices in the order found by Dijkstra's Algorithm

	// Set the start time of Brandes' Algorithm

	// Compute Betweenness Centrality for every vertex i
	for (i=0; i < nvertices; i++)
	{
		/* Initialize */
		PredList.assign(nvertices, std::vector <int> (0, 0));
		d.assign(nvertices, Long_max);
		d[i] = 0;
		sigma.assign(nvertices, 0);
		sigma[i] = 1;
		delta.assign(nvertices, 0);
		Q.push(i);

		// Use Breadth First Search algorithm
		while (!Q.empty()) {
			// Get the next element in the queue
			u = Q.front();
			Q.pop();
			// Push u onto the stack S. Needed later for betweenness computation
			S.push(u);
			// Iterate over all the neighbors of u
			for (j=0; j < (int) _g->adj_list[u].size(); j++) {
				// Get the neighbor v of vertex u
				// v = (ui64) network->vertex[u].edge[j].target;
				v = (int) _g->adj_list[u][j];

				/* Relax and Count */
				if (d[v] == Long_max) {
					 d[v] = d[u] + 1;
					 Q.push(v);
				}
				if (d[v] == d[u] + 1) {
					sigma[v] += sigma[u];
					PredList[v].push_back(u);
				}
			} // End For

		} // End While

		/* Accumulation */
		while (!S.empty()) {
			u = S.top();
			S.pop();
			for (j=0; j < PredList[u].size(); j++) {
				delta[PredList[u][j]] += ((double) sigma[PredList[u][j]]/sigma[u]) * (1+delta[u]);
			}
			if (u != i)
				CB[u] += delta[u];
		}

		// Clear data for the next run
		PredList.clear();
		d.clear();
		sigma.clear();
		delta.clear();
	} // End For

	// End time after Brandes' algorithm and the time difference

    for(int i =0; i<nvertices;++i){
        CB[i]=CB[i]/norm;
        if(CB[i]==0){
            bc_bool[i] = 0;
            bc_log[i] = 0.0;
        }
        else{
            bc_bool[i] = 1;
            bc_log[i]=-log10(CB[i]);
        }
    }

	return CB;

} //


std::vector<double> Utils::convertToLog(std::vector<double> CB)
{
    std::vector<double> result;
    for(int i =0;i<CB.size();++i)
    {
        if(CB[i]==0)
        {
            result.push_back(0.0);
        }
        else
        {
            result.push_back(-log10(CB[i]));
        }
    }
    return result;
}

