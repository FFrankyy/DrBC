# DrBC
This is a TensorFlow implementation of Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach, as described in our paper:

Fan, Changjun and Zeng, Li and Ding, Yuhui and Chen, Muhao and Sun, Yizhou and Liu, Zhong[Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach](http://arxiv.org/abs/1905.10418) (CIKM 2019)

![](./visualize/Figure_demo.jpg "Demo")


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
