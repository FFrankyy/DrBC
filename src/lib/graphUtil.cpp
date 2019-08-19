#include "graphUtil.h"

graphUtil::graphUtil(int totalSize) : startIt(0),endIt(1),node_num(0),degree_sum(0)
{
    bfsQueue.resize(totalSize, -1);
    alreadyAccessBool.resize(totalSize, 0);
}

std::vector<int>::iterator graphUtil::getBeginItForBfsQueue()
{
    return bfsQueue.begin();
}

std::vector<int>::iterator graphUtil::getEndItForBfsQueue()
{
    return bfsQueue.begin() + endIt;
}

void graphUtil::getNeighbourFrontierAndScope(const std::vector<std::vector<int> > &adjListGraph, int scope, int currentNode)
{
    startIt = 0;
    endIt = 1;

    bfsQueue[0] = currentNode;
    alreadyAccessBool[currentNode] = 1;

    for (int i = 0; i < scope; i++)
    {
        int lastEndIt = endIt;
        while (startIt != lastEndIt)
        {
            const std::vector<int>& neighbourNodeList = adjListGraph[bfsQueue[startIt++]];

            for (const auto& eachNeighbour : neighbourNodeList)
            {
                if (!alreadyAccessBool[eachNeighbour])
                {
                    bfsQueue[endIt++] = eachNeighbour;
                    alreadyAccessBool[eachNeighbour] = 1;
                }
            }
        }
    }

    for (int i = 0; i < endIt; i++)
    {
        alreadyAccessBool[bfsQueue[i]] = 0;
    }
}



long long graphUtil::basicCi(const std::vector<std::vector<int> > &adjListGraph, int ballRadius, int currentNode)
{
    if (adjListGraph[currentNode].size() == 0)
    {
        return -1;
    }

    if (ballRadius == 0)
    {
        return adjListGraph[currentNode].size();
    }

    getNeighbourFrontierAndScope(adjListGraph, ballRadius, currentNode);

    long long ci = 0;

    for (int i = startIt; i < endIt; i++)
    {
        ci += (adjListGraph[bfsQueue[i]].size() - 1);
    }
    degree_sum = ci;
    node_num = endIt - startIt;

    ci *= (adjListGraph[currentNode].size() - 1);

    return ci;
}
