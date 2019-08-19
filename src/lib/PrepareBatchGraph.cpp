#include "PrepareBatchGraph.h"
#define max(x, y) (x > y ? x : y)
#define min(x, y) (x > y ? y : x)

sparseMatrix::sparseMatrix()
{
    rowNum = 0;
    colNum = 0;
}

sparseMatrix::~sparseMatrix()
{
    rowNum = 0;
    colNum = 0;
    rowIndex.clear();
    colIndex.clear();
    value.clear();
}

 PrepareBatchGraph::PrepareBatchGraph(int _aggregatorID)
{
    aggregatorID = _aggregatorID;
}

PrepareBatchGraph::~PrepareBatchGraph()
{
    n2nsum_param =nullptr;
    subgsum_param =nullptr;
    aux_feat.clear();
    node_feat.clear();
    neighbor_param.clear();
    idx_map_list.clear();
    size_list.clear();
    aggregatorID = -1;
}


void PrepareBatchGraph::SetupBatchGraph(std::vector< std::shared_ptr<Graph> > g_list)
{

    int node_cnt = 0;

    idx_map_list.resize(g_list.size());
    size_list.resize(g_list.size());

    for (size_t i = 0; i < g_list.size(); ++i)
    {
        auto g = g_list[i];
        node_cnt += g->num_nodes;
    }

    graph.Resize(g_list.size(), node_cnt);
    node_cnt = 0;
    int edge_cnt = 0;

    Utils myUtils = Utils();

    for (size_t i = 0; i < g_list.size(); ++i)
    {

        auto g = g_list[i];

        std::vector<int> idx_map;
        idx_map.resize(g->num_nodes);
        graphUtil gu = graphUtil(g->num_nodes);
        int t = 0;

        std::vector<int>::iterator it;

        std::vector<double> ccList;
        std::vector<double> dcList;
        std::vector<double> CI1_List;
        std::vector<double> CI2_List;
        int maxDegree = 0;
        int totalDegree = 0;


        for(int j = 0; j < g->num_nodes; ++j)
        {
            int degree = g->adj_list[j].size();

            totalDegree += degree;
            if(degree>maxDegree)
            {
                maxDegree = degree;
            }
            dcList.push_back((double)degree/(double)g->num_nodes);
        }

        for(int j = 0; j < g->num_nodes; ++j)
        {
            idx_map[j] = 0;
            int degree = g->adj_list[j].size();
            int Norm = degree*(degree-1);

            double cluster_coefficient = 0.0;

            if(degree>1)
            {
                int cc = 0;
                for (auto k:g->adj_list[j])
                {
                    for (auto m:g->adj_list[j])
                    {
                        it=find(g->adj_list[k].begin(),g->adj_list[k].end(),m);
                        if (it!=g->adj_list[k].end())
                        {
                            cc += 1;
                        }
                    }
                }
                cluster_coefficient = cc/Norm;
            }

            if(cluster_coefficient >=1.0||degree<=1)
            {
                idx_map[j] = -1;
            }
            ccList.push_back(cluster_coefficient);

            long long CI1 = gu.basicCi(g->adj_list,1,j);
            double ci1=(double)CI1/(double)(maxDegree*totalDegree);

            long long CI2 = gu.basicCi(g->adj_list,2,j);
            double ci2=(double)CI2/(double)(maxDegree*totalDegree);

            CI1_List.push_back(ci1);
            CI2_List.push_back(ci2);
        }

        for (int j = 0; j < g->num_nodes; ++j)
        {


            std::vector<double> per_node_feat;
            std::vector<double> per_aux_feat;
            // degree
            per_node_feat.push_back(dcList[j]);
            per_aux_feat.push_back(dcList[j]);

            // CI1
            per_aux_feat.push_back(CI1_List[j]);

            // CI2
            per_aux_feat.push_back(CI2_List[j]);

            per_node_feat.push_back(1.0);
            per_node_feat.push_back(1.0);

            per_aux_feat.push_back(1.0);

            node_feat.push_back(per_node_feat);
            aux_feat.push_back(per_aux_feat);

            graph.AddNode(i, node_cnt + t);

            t += 1;
        }

        idx_map_list[i] = idx_map;
        size_list[i] = t;
        assert(t == g->num_nodes);
        for (auto p : g->edge_list)
        {
            auto x = p.first + node_cnt, y = p.second + node_cnt;
            graph.AddEdge(edge_cnt, x, y);
            edge_cnt += 1;
            graph.AddEdge(edge_cnt, y, x);
            edge_cnt += 1;
        }
        node_cnt += g->num_nodes;
    }
    assert(node_cnt == (int)graph.num_nodes);

    if(aggregatorID<=2)
    {
        n2nsum_param = n2n_construct(&graph,aggregatorID);
    }

}


bool cmp(std::pair<int, int>a, std::pair<int, int>b)
{
    return a.second>b.second;
}


std::shared_ptr<sparseMatrix> n2n_construct(GraphStruct* graph,int aggregatorID)
{
    //aggregatorID = 0 sum
    //aggregatorID = 1 mean
    //aggregatorID = 2 GCN
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_nodes;

	for (unsigned int i = 0; i < graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];

		for (size_t j = 0; j < list.size(); ++j)
		{
		    switch(aggregatorID)
		    {
		       case 0:
		       {
		          result->value.push_back(1.0);
		          break;
		       }
		       case 1:
		       {
		          result->value.push_back(1.0/(double)list.size());
		          break;
		       }
		       case 2:
		       {
		          int neighborDegree = (int)graph->in_edges->head[list[j].second].size();
		          int selfDegree = (int)list.size();
		          double norm = sqrt((double)(neighborDegree+1))*sqrt((double)(selfDegree+1));
		          result->value.push_back(1.0/norm);
		          break;
		       }
		       default:
		       {
		          result->value.push_back(1.0);
		          break;
		       }
		    }
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].second);
		}
	}
    return result;
}

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_edges;

	for (unsigned int i = 0; i < graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);
		}
	}
    return result;
}



std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_nodes;
	for (unsigned int i = 0; i < graph->num_edges; ++i)
	{
        result->value.push_back(1.0);
        result->rowIndex.push_back(i);
        result->colIndex.push_back(graph->edge_list[i].first);
	}
    return result;
}

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_edges;
    for (unsigned int i = 0; i < graph->num_edges; ++i)
    {
        int node_from = graph->edge_list[i].first, node_to = graph->edge_list[i].second;
        auto& list = graph->in_edges->head[node_from];
        for (size_t j = 0; j < list.size(); ++j)
        {
            if (list[j].second == node_to)
                continue;
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);
        }
    }
    return result;
}

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph, std::vector<std::pair<int,int>>& subgraph_id_span)
{
   std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_subgraph;
    result->colNum = graph->num_nodes;

    subgraph_id_span.clear();
    int start = 0;
    int end = 0;
	for (unsigned int i = 0; i < graph->num_subgraph; ++i)
	{

		auto& list = graph->subgraph->head[i];
        end  = start + list.size() - 1;
		for (size_t j = 0; j < list.size(); ++j)
		{
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j]);
		}
		if(list.size()>0){
		    subgraph_id_span.push_back(std::make_pair(start,end));
		}
		else{
		    subgraph_id_span.push_back(std::make_pair(graph->num_nodes,graph->num_nodes));
		}
		start = end +1 ;
	}
    return result;
}

