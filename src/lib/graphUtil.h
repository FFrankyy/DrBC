#ifndef GRAPHUTIL_H
#define GRAPHUTIL_H

#include <vector>
#include <iterator>
#include <algorithm>
#include <set>
#include <memory>


class graphUtil
{
public:
	std::vector<bool> alreadyAccessBool;
	std::vector<int> bfsQueue;
	int startIt;
	int endIt;
	int node_num;
	int degree_sum;
	graphUtil(int totalSize);
	std::vector<int>::iterator getBeginItForBfsQueue();
	std::vector<int>::iterator getEndItForBfsQueue();
	void getNeighbourFrontierAndScope(const std::vector<std::vector<int> > &adjListGraph, int scope, int currentNode);
	long long basicCi(const std::vector<std::vector<int> > &adjListGraph, int ballRadius, int currentNode);
};

#endif