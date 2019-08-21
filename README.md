# DrBC
This is a TensorFlow implementation of 'Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach', as described in our paper:

Fan, Changjun and Zeng, Li and Ding, Yuhui and Chen, Muhao and Sun, Yizhou and Liu, Zhong[[Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach]](http://arxiv.org/abs/1905.10418) (CIKM 2019)

![](./visualize/Figure_demo.jpg "Demo")

This code folder is organized as follows:

+ __models/__: contains the model to obtain the results in the paper
+ __src/__: set of c source codes used in the paper
+ c source files:
    + /__src__/__lib__/__PrepareBatchGraph.cpp__: functions for Prepare the batch graphs used in the tensorflow codes.
    + /__src__/__lib__/__graph.cpp__: basic structure for graphs.
    + /__src__/__lib__/__graphUtil.cpp__: functions for computing the collective influence functions.
    + /__src__/__lib__/__graph_struct.cpp__: Linked list data structure for sparse graphs.
    + /__src__/__lib__/__metrics.cpp__: functions for computing the metrics functions such as MeanSquareError, AvgError and RankTopK. 
    + /__src__/__lib__/__utils.cpp__: functions for computing the Betweenness functions.
+ __visualize/__: contains the figures used in the paper

In order to make our program more efficient,we write C extensions for Python based on [Cython](https://cython.org/) which is an optimising static compiler for Python programming language, and the bindings files for c codes are listed as follows:
+  cython files:
    + /__PrepareBatchGraph.pyx__: Cython bindings of PrepareBatchGraph.cpp.
    + /__PrepareBatchGraph.pxd__: Header file of PrepareBatchGraph.pyx.
    + /__graph.pyx__: Cython bindings of graph.cpp.
    + /__graph.pxd__: Header file of graph.pyx.
    + /__graphUtil.pyx__: Cython bindings of graphUtil.cpp.
    + /__graphUtil.pxd__: Header file of graphUtil.pyx.
    + /__graph_struct.pyx__: Cython bindings of graph_struct.cpp.
    + /__graph_struct.pxd__: header file of graph_struct.pyx.  
    + /__metrics.pyx__: Cython bindings of metrics.cpp.
    + /__metrics.pxd__: Header file of metrics.pyx.   
    + /__utils.pyx__: Cython bindings of utils.cpp.
    + /__utils.pxd__: Header file of utils.pyx. 

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
