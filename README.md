# DrBC
Implementation of 'Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach'(https://arxiv.org/pdf/1905.10418.pdf)

![](./visualize/Figure_demo.jpg "Demo")


# 1. Build
Get the source code, and install all the dependencies.
```ruby
git clone https://github.com/FFrankyy/DrBC.git
pip install requirements.txt
```

makefile
```ruby
python setup.py build_ext -i
```

# 2.Training
adjust hyper-parameters in BetLearn.py, and run the following to train the model
```ruby
python start.py
```


# 3.Reproducing the results that reported in the paper
Here is the link to the dataset that was used in the paper:
```ruby
https://drive.google.com/file/d/1nh9XRyrqtKsaBDpLJri-SotpU3f713SX/view?usp=sharing
```

# 4.Baselines implementations
For ABRA, RK and k-BC, we use the following implementations:
```ruby
https://github.com/ecrc/BeBeCA
```
For KADABRA, we use the following implementations:
```ruby
https://github.com/natema/kadabra
```

# 4.Reference
Please cite our work if you find our code/paper is useful to your work.

```ruby
@article{fan2019learning,
  title={Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach},
  author={Fan, Changjun and Zeng, Li and Ding, Yuhui and Chen, Muhao and Sun, Yizhou and Liu, Zhong},
  journal={arXiv preprint arXiv:1905.10418},
  year={2019}
}
```
