# DrBC
This is a TensorFlow implementation of 'Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach', as described in our paper:

Fan, Changjun and Zeng, Li and Ding, Yuhui and Chen, Muhao and Sun, Yizhou and Liu, Zhong[[Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach]](http://arxiv.org/abs/1905.10418) (CIKM 2019)

![](./visualize/Figure_demo.jpg "Demo")

This code folder is organized as follows:

+ __models/__: contains the model to obtain the results in the paper
+ __src/__: set of c source codes used in the paper
+ files:
    + /__lib__/__PrepareBatchGraph.cpp__: functions for Prepare the batch graphs used in the tensorflow codes.
    + /__lib__/__graph.cpp__: basic structure for graphs.
    + /__lib__/__graphUtil.cpp__: functions for computing the collective influence functions.
    + /__lib__/__graph_struct.cpp__: Linked list data structure for sparse graphs.
    + /__lib__/__metrics.cpp__: functions for computing the metrics functions such as MeanSquareError, AvgError, MaxError, RankTopK. 
    + /__lib__/__utils.cpp.cpp__: functions for computing the Betweenness functions.
+ __visualize/__: contains the figures uesed in the paper


# 1. Build
Get the source code, and install all the dependencies.
```
git clone https://github.com/FFrankyy/DrBC.git
pip install -r requirements.txt
```

Makefile
```
python setup.py build_ext -i
```

# 2.Training
Adjust hyper-parameters in BetLearn.py, and run the following to train the model
```
python start.py
```


# 3.Reproducing the results that reported in the paper
Here is the link to the dataset that was used in the paper:
```
https://drive.google.com/file/d/1nh9XRyrqtKsaBDpLJri-SotpU3f713SX/view?usp=sharing
```
The model to obtain the results in the paper is in the fold './models/'

# 4.Baselines implementations
For RK and k-BC, we use the following implementations:
```
https://github.com/ecrc/BeBeCA
```
For KADABRA, we use:
```
https://github.com/natema/kadabra
```
For ABRA, we use the codes in the original paper.
For node2vec, we use:
```
https://github.com/snap-stanford/snap/tree/master/examples/node2vec
```

# 4.Reference
Please cite our work if you find our code/paper is useful to your work.

```
@article{fan2019learning,
  title={Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach},
  author={Fan, Changjun and Zeng, Li and Ding, Yuhui and Chen, Muhao and Sun, Yizhou and Liu, Zhong},
  journal={arXiv preprint arXiv:1905.10418},
  year={2019}
}
```
