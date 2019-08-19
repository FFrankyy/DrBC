#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: fanchangjun
"""
from __future__ import print_function, division
import tensorflow as tf
import networkx as nx
import time
import sys
import numpy as np
from tqdm import tqdm
import graph
import utils
import PrepareBatchGraph
import metrics
import pickle as cp
import os


EMBEDDING_SIZE = 128 # embedding dimension
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
max_bp_iter = 5     # neighbor propagation steps

REG_HIDDEN = (int)(EMBEDDING_SIZE / 2) # hidden dimension in the  MLP decoder
initialization_stddev = 0.01
NUM_MIN = 100  # minimum training scale (node set size)
NUM_MAX = 200  # maximum training scale (node set size)
MAX_ITERATION = 10000   # training iterations
n_valid = 100   # number of validation graphs
aggregatorID = 2 # how to aggregate node neighbors, 0:sum; 1:mean; 2:GCN(weighted sum)
combineID = 1   # how to combine self embedding and neighbor embedding,
                   # 0:structure2vec(add node feature and neighbor embedding)
                   #1:graphsage(concatenation); 2:gru
JK = 1  # layer aggregation,  #0: do not use each layer's embedding;
                           #aggregate each layer's embedding with:
                           # 1:max_pooling; 2:min_pooling;
                           # 3:mean_pooling; 4:LSTM with attention
node_feat_dim = 3  # initial node features, [Dc,1,1]
aux_feat_dim = 4   # extra node features in the hidden layer in the decoder, [Dc,CI1,CI2,1]

INF = 100000000000

class BetLearn:

    def __init__(self):
        # init some parameters
        self.g_type = 'powerlaw' #'erdos_renyi', 'powerlaw', 'small-world', 'barabasi_albert'
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.reg_hidden = REG_HIDDEN
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.utils = utils.py_Utils()
        self.TrainBetwList = []
        self.TestBetwList = []
        self.metrics = metrics.py_Metrics()
        self.inputs = dict()
        self.activation = tf.nn.leaky_relu   #leaky_relu relu selu elu

        self.ngraph_train = 0
        self.ngraph_test = 0

        # [node_cnt, node_feat_dim]
        self.node_feat = tf.placeholder(tf.float32, name="node_feat")
        # [node_cnt, aux_feat_dim]
        self.aux_feat = tf.placeholder(tf.float32, name="aux_feat")
        # [node_cnt, node_cnt]
        self.n2nsum_param = tf.sparse_placeholder(tf.float64, name="n2nsum_param")


        # [node_cnt,1]
        self.label = tf.placeholder(tf.float32, shape=[None,1], name="label")
        # sample node pairs to compute the ranking loss
        self.pair_ids_src = tf.placeholder(tf.int32, shape=[1,None], name='pair_ids_src')
        self.pair_ids_tgt = tf.placeholder(tf.int32, shape=[1,None], name='pair_ids_tgt')

        self.loss, self.trainStep, self.betw_pred, self.node_embedding, self.param_list = self.BuildNet()

        self.saver = tf.train.Saver(max_to_keep=None)
        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=100,
                                intra_op_parallelism_threads=100,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self.session.run(tf.global_variables_initializer())


    def BuildNet(self):
        # [node_feat_dim, embed_dim]
        w_n2l = tf.Variable(tf.truncated_normal([node_feat_dim, self.embedding_size], stddev=initialization_stddev), tf.float32, name="w_n2l")
        # [embed_dim, embed_dim]
        p_node_conv = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="p_node_conv")

        if combineID == 1:  # 'graphsage'
            # [embed_dim, embed_dim]
            p_node_conv2 = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="p_node_conv2")
            # [2*embed_dim, embed_dim]
            p_node_conv3 = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="p_node_conv3")
        elif combineID ==2: #GRU
            w_r = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="w_r")
            u_r = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="u_r")
            w_z = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="w_z")
            u_z = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="u_z")
            w = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="w")
            u = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="u")

        # [embed_dim, reg_hidden]
        h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32,name="h1_weight")
        # [reg_hidden+aux_feat_dim, 1]
        h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden+aux_feat_dim, 1], stddev=initialization_stddev), tf.float32,name="h2_weight")
        # [reg_hidden, 1]
        last_w = h2_weight

        # [node_cnt, node_feat_dim]
        node_size = tf.shape(self.n2nsum_param)[0]
        node_input = self.node_feat

        #[node_cnt, embed_dim]
        input_message = tf.matmul(tf.cast(node_input, tf.float32), w_n2l)

        lv = 0
        # [node_cnt, embed_dim], no sparse
        cur_message_layer = self.activation(input_message)
        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

        if JK:  # # 1:max_pooling; 2:min_pooling; 3:mean_pooling; 4:LSTM with attention
            cur_message_layer_JK = cur_message_layer
        if JK == 4:  #LSTM init hidden layer
            w_r_JK = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="w_r_JK")
            u_r_JK = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="u_r_JK")
            w_z_JK = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="w_z_JK")
            u_z_JK = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="u_z_JK")
            w_JK = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="w_JK")
            u_JK = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32,name="u_JK")
            #attention matrix
            JK_attention = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32,name="JK_attention")
            #attention list
            JK_attention_list =[]
            JK_Hidden_list=[]
            cur_message_layer_list = []
            cur_message_layer_list.append(cur_message_layer)
            JK_Hidden = tf.truncated_normal(tf.shape(cur_message_layer), stddev=initialization_stddev)

        # max_bp_iter steps of neighbor propagation
        while lv < max_bp_iter:
            lv = lv + 1
            # [node_cnt, node_cnt]*[node_cnt, embed_dim] = [node_cnt, embed_dim]
            n2npool = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param, tf.float64), tf.cast(cur_message_layer, tf.float64))
            n2npool = tf.cast(n2npool, tf.float32)

            # [node_cnt, embed_dim] * [embedding, embedding] = [node_cnt, embed_dim], dense
            node_linear = tf.matmul(n2npool, p_node_conv)

            if combineID == 0:  # 'structure2vec'
                # [node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
                merged_linear = tf.add(node_linear, input_message)
                # [node_cnt, embed_dim]
                cur_message_layer = self.activation(merged_linear)
                if JK==1:
                    cur_message_layer_JK = tf.maximum(cur_message_layer_JK,cur_message_layer)
                elif JK==2:
                    cur_message_layer_JK = tf.minimum(cur_message_layer_JK, cur_message_layer)
                elif JK==3:
                    cur_message_layer_JK = tf.add(cur_message_layer_JK, cur_message_layer)
                elif JK == 4:
                    cur_message_layer_list.append(cur_message_layer)

            elif combineID == 1:  # 'graphsage'
                # [node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                cur_message_layer_linear = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2)
                # [[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                merged_linear = tf.concat([node_linear, cur_message_layer_linear], 1)
                # [node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                cur_message_layer = self.activation(tf.matmul(merged_linear, p_node_conv3))

                if JK == 1:
                    cur_message_layer_JK = tf.maximum(cur_message_layer_JK,cur_message_layer)
                elif JK == 2:
                    cur_message_layer_JK = tf.minimum(cur_message_layer_JK, cur_message_layer)
                elif JK == 3:
                    cur_message_layer_JK = tf.add(cur_message_layer_JK, cur_message_layer)
                elif JK == 4:
                    cur_message_layer_list.append(cur_message_layer)

            elif combineID==2: #gru
                r_t = tf.nn.relu(tf.add(tf.matmul(node_linear,w_r), tf.matmul(cur_message_layer,u_r)))
                z_t = tf.nn.relu(tf.add(tf.matmul(node_linear,w_z), tf.matmul(cur_message_layer,u_z)))
                h_t = tf.nn.tanh(tf.add(tf.matmul(node_linear,w), tf.matmul(r_t*cur_message_layer,u)))
                cur_message_layer = (1-z_t)*cur_message_layer + z_t*h_t
                cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

                if JK == 1:
                    cur_message_layer_JK = tf.maximum(cur_message_layer_JK,cur_message_layer)
                elif JK == 2:
                    cur_message_layer_JK = tf.minimum(cur_message_layer_JK, cur_message_layer)
                elif JK == 3:
                    cur_message_layer_JK = tf.add(cur_message_layer_JK, cur_message_layer)
                elif JK == 4:
                    cur_message_layer_list.append(cur_message_layer)

            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

        if JK == 1 or JK == 2:
            cur_message_layer = cur_message_layer_JK
        elif JK == 3:
            cur_message_layer = cur_message_layer_JK / (max_bp_iter+1)
        elif JK == 4:
            for X_value in cur_message_layer_list:
                #[node_cnt,embed_size]
                r_t_JK = tf.nn.relu(tf.add(tf.matmul(X_value, w_r_JK), tf.matmul(JK_Hidden, u_r_JK)))
                z_t_JK = tf.nn.relu(tf.add(tf.matmul(X_value, w_z_JK), tf.matmul(JK_Hidden, u_z_JK)))
                h_t_JK = tf.nn.tanh(tf.add(tf.matmul(X_value, w_JK), tf.matmul(r_t_JK * JK_Hidden, u_JK)))
                JK_Hidden = (1 - z_t_JK) * h_t_JK + z_t_JK * JK_Hidden
                JK_Hidden = tf.nn.l2_normalize(JK_Hidden, axis=1)
                #[max_bp_iter+1,node_cnt,embed_size]
                JK_Hidden_list.append(JK_Hidden)
                # [max_bp_iter+1,node_cnt,1] =  [node_cnt,embed_size]*[embed_size,1]=[node_cnt,1]
                attention = tf.nn.tanh(tf.matmul(JK_Hidden, JK_attention))
                JK_attention_list.append(attention)
                cur_message_layer = JK_Hidden

            # [max_bp_iter+1,node_cnt,1]
            JK_attentions = tf.reshape(JK_attention_list, [max_bp_iter + 1, node_size, 1])
            cofficient = tf.nn.softmax(JK_attentions, axis=0)
            JK_Hidden_list = tf.reshape(JK_Hidden_list, [max_bp_iter + 1, node_size, self.embedding_size])
            # [max_bpr_iter+1,node_cnt,1]* [max_bp_iter + 1,node_cnt,embed_size] = [max_bp_iter + 1,node_cnt,embed_size]
            #[max_bp_iter + 1,node_cnt,embed_size]
            result = cofficient * JK_Hidden_list
            cur_message_layer = tf.reduce_sum(result, 0)
            cur_message_layer = tf.reshape(cur_message_layer, [node_size, self.embedding_size])

        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

        # node embedding, [node_cnt, embed_dim]
        embed_s_a = cur_message_layer

        # decoder, two-layer MLP
        hidden = tf.matmul(embed_s_a, h1_weight)
        last_output = self.activation(hidden)
        last_output = tf.concat([last_output, self.aux_feat], axis=1)
        betw_pred = tf.matmul(last_output, last_w)

        # [pair_size, 1]
        labels = tf.nn.embedding_lookup(self.label, self.pair_ids_src) - tf.nn.embedding_lookup(self.label, self.pair_ids_tgt)
        preds = tf.nn.embedding_lookup(betw_pred, self.pair_ids_src) - tf.nn.embedding_lookup(betw_pred, self.pair_ids_tgt)

        loss = self.pairwise_ranking_loss(preds, labels)
        trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return loss, trainStep, betw_pred,embed_s_a,tf.trainable_variables()

    def pairwise_ranking_loss(self, preds, labels):
        """Logit cross-entropy loss with masking."""
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=tf.sigmoid(labels))
        loss = tf.reduce_sum(loss, axis=1)
        return tf.reduce_mean(loss)

    def gen_graph(self, num_min, num_max):
        cur_n = np.random.randint(num_max - num_min + 1) + num_min
        if self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=4)
        elif self.g_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        return g

    def gen_new_graphs(self, num_min, num_max):
        print('\ngenerating new training graphs...')
        self.ClearTrainGraphs()
        for i in tqdm(range(1000)):
            g = self.gen_graph(num_min, num_max)
            self.InsertGraph(g, is_test=False)
            bc = self.utils.Betweenness(self.GenNetwork(g))
            bc_log = self.utils.bc_log
            self.TrainBetwList.append(bc_log)

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()
        self.TrainBetwList = []
        self.TrainBetwRankList = []

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()
        self.TestBetwList = []

    def InsertGraph(self, g, is_test):
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork(g))
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork(g))

    def PrepareValidData(self):
        print('\ngenerating validation graphs...')
        sys.stdout.flush()
        self.ClearTestGraphs()
        for i in tqdm(range(n_valid)):
            g = self.gen_graph(NUM_MIN, NUM_MAX)
            self.InsertGraph(g, is_test=True)
            bc = self.utils.Betweenness(self.GenNetwork(g))
            self.TestBetwList.append(bc)

    def SetupBatchGraph(self,g_list):
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupBatchGraph(g_list)
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['node_feat'] = prepareBatchGraph.node_feat
        self.inputs['aux_feat'] = prepareBatchGraph.aux_feat
        self.inputs['pair_ids_src'] = prepareBatchGraph.pair_ids_src
        self.inputs['pair_ids_tgt'] = prepareBatchGraph.pair_ids_tgt
        assert (len(prepareBatchGraph.pair_ids_src) == len(prepareBatchGraph.pair_ids_tgt))
        return prepareBatchGraph.idx_map_list

    def SetupTrain(self, g_list, label_log):
        self.inputs['label'] = label_log
        self.SetupBatchGraph(g_list)

    def SetupPred(self, g_list):
        idx_map_list = self.SetupBatchGraph(g_list)
        return idx_map_list

    def Predict(self, g_list):
        idx_map_list = self.SetupPred(g_list)
        my_dict=dict()
        my_dict[self.n2nsum_param]=self.inputs['n2nsum_param']
        my_dict[self.aux_feat] = self.inputs['aux_feat']
        my_dict[self.node_feat] = self.inputs['node_feat']
        result = self.session.run([self.betw_pred], feed_dict=my_dict)

        idx_map = idx_map_list[0]
        result_output = []
        result_data = result[0]
        for i in range(len(result_data)):
            if idx_map[i] >= 0: # corresponds to nodes with 0.0 betw_log value
                result_output.append(np.power(10,-result_data[i][0]))
            else:
                result_output.append(0.0)
        return result_output

    def Fit(self):
        g_list, id_list = self.TrainSet.Sample_Batch(BATCH_SIZE)
        Betw_Label_List = []

        for id in id_list:
            Betw_Label_List += self.TrainBetwList[id]
        label = np.resize(Betw_Label_List, [len(Betw_Label_List), 1])
        self.SetupTrain(g_list, label)

        my_dict=dict()
        my_dict[self.n2nsum_param]=self.inputs['n2nsum_param']
        my_dict[self.aux_feat] = self.inputs['aux_feat']
        my_dict[self.node_feat] = self.inputs['node_feat']
        my_dict[self.label] = self.inputs['label']
        my_dict[self.pair_ids_src] = np.reshape(self.inputs['pair_ids_src'], [1, len(self.inputs['pair_ids_src'])])
        my_dict[self.pair_ids_tgt] = np.reshape(self.inputs['pair_ids_tgt'], [1, len(self.inputs['pair_ids_tgt'])])
        result = self.session.run([self.loss, self.trainStep], feed_dict=my_dict)

        loss = result[0]
        return loss / len(g_list)

    def Train(self):
        self.PrepareValidData()
        self.gen_new_graphs(NUM_MIN, NUM_MAX)

        save_dir = './models'
        VCFile = '%s/ValidValue.csv' % (save_dir)
        f_out = open(VCFile, 'w')
        for iter in range(MAX_ITERATION):
            TrainLoss = self.Fit()
            start = time.clock()
            if iter and iter % 5000 == 0:
                self.gen_new_graphs(NUM_MIN, NUM_MAX)
            if iter % 500 == 0:
               if (iter == 0):
                   N_start = start
               else:
                   N_start = N_end
               frac_topk, frac_kendal = 0.0, 0.0
               test_start = time.time()
               for idx in range(n_valid):
                   run_time, temp_topk, temp_kendal = self.Test(idx)
                   frac_topk += temp_topk / n_valid
                   frac_kendal += temp_kendal / n_valid
               test_end = time.time()
               f_out.write('%.6f, %.6f\n' %(frac_topk, frac_kendal))  # write vc into the file
               f_out.flush()
               print('\niter %d, Top0.01: %.6f, kendal: %.6f'%(iter, frac_topk, frac_kendal))
               print('testing %d graphs time: %.2fs' % (n_valid, test_end - test_start))
               N_end = time.clock()
               print('500 iterations total time: %.2fs' % (N_end - N_start))
               print('Training loss is %.4f' % TrainLoss)
               sys.stdout.flush()
               model_path = '%s/nrange_iter_%d_%d_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX,iter)
               self.SaveModel(model_path)
        f_out.close()

    def Test(self, gid):
        g_list = [self.TestSet.Get(gid)]
        start = time.time()
        betw_predict = self.Predict(g_list)
        end = time.time()
        betw_label = self.TestBetwList[gid]

        run_time = end - start
        topk = self.metrics.RankTopK(betw_label,betw_predict, 0.01)
        kendal = self.metrics.RankKendal(betw_label,betw_predict)
        return run_time, topk, kendal

    def findModel(self):
        VCFile = './models/ValidValue.csv'
        vc_list = []
        EarlyStop_start = 2
        EarlyStop_length = 1
        num_line = 0
        for line in open(VCFile):
            data = float(line.split(',')[0].strip(','))    #0:topK; 1:kendal
            vc_list.append(data)
            num_line += 1
            if num_line > EarlyStop_start and data < np.mean(vc_list[-(EarlyStop_length+1):-1]):
                best_vc = num_line
                break
        best_model_iter = 500 * best_vc
        best_model = './models/nrange_iter_%d.ckpt' % (best_model_iter)
        return best_model

    def EvaluateSynData(self, data_test, model_file=None):  # test synthetic data
        if model_file == None:  # if user do not specify the model_file
            model_file = self.findModel()
        print('The best model is :%s' % (model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        n_test = 100
        frac_run_time, frac_topk, frac_kendal = 0.0, 0.0, 0.0
        self.ClearTestGraphs()
        f = open(data_test, 'rb')
        ValidData = cp.load(f)
        TestGraphList = ValidData[0]
        self.TestBetwList = ValidData[1]
        for i in tqdm(range(n_test)):
            g = TestGraphList[i]
            self.InsertGraph(g, is_test=True)
            run_time, topk, kendal = self.test(i)
            frac_run_time += run_time/n_test
            frac_topk += topk/n_test
            frac_kendal += kendal/n_test
        print('\nRun_time, Top1%, Kendall tau: %.6f, %.6f, %.6f'% (frac_run_time, frac_topk, frac_kendal))
        return frac_run_time, frac_topk, frac_kendal


    def EvaluateRealData(self, model_file, data_test, label_file):  # test real data
        g = nx.read_weighted_edgelist(data_test)
        sys.stdout.flush()
        self.LoadModel(model_file)
        betw_label = []
        for line in open(label_file):
            betw_label.append(float(line.strip().split()[1]))
        self.TestBetwList.append(betw_label)
        start = time.time()
        self.InsertGraph(g, is_test=True)
        end = time.time()
        run_time = end - start
        g_list = [self.TestSet.Get(0)]
        start1 = time.time()
        betw_predict = self.Predict(g_list)
        end1 = time.time()
        betw_label = self.TestBetwList[0]
        run_time += end1 - start1
        top001 = self.metrics.RankTopK(betw_label, betw_predict, 0.01)
        top005 = self.metrics.RankTopK(betw_label, betw_predict, 0.05)
        top01 = self.metrics.RankTopK(betw_label, betw_predict, 0.1)
        kendal = self.metrics.RankKendal(betw_label, betw_predict)
        self.ClearTestGraphs()
        return top001, top005, top01, kendal, run_time


    def SaveModel(self, model_path):
        self.saver.save(self.session, model_path)
        print('model has been saved success!')

    def LoadModel(self, model_path):
        self.saver.restore(self.session, model_path)
        print('restore model from file successfully')

    def GenNetwork(self, g):  # networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
        else:
            A = np.array([0])
            B = np.array([0])
        return graph.py_Graph(len(g.nodes()), len(edges), A, B)