# -*- coding: utf-8 -*-
# @Time    : 2019/5/9 8:24
# @Author  : Weiyang
# @File    : TrainModel.py
'''
训练模型
'''
import tensorflow as tf
from pathlib import Path
import functools
import numpy as np
from tf_metrics import precision,recall,f1
import logging
import sys

class TrainModel(object):

    def __init__(self):
        self.cfg = self.__getConfig()  # configparser.ConfigParser().read()返回的对象
        self.estimator = None # 存储训练好的模型

    def __getConfig(self):
        '读取配置'
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read('../config/config.ini',encoding='utf-8')
        return cfg

    def __parse_fn(self,line_words,line_tags):
        '读取一行words和其对应的一行tags,转换编码为bytes'
        'line_words: word1 word2 word3 ... ; line_tags: tag1 tag2 tag3 ...'
        words = [w.encode() for w in line_words.strip().split()]
        tags = [t.encode() for t in line_tags.strip().split()]
        assert len(words) == len(tags),"单词和标记数量不匹配"
        return (words,len(words)),tags # ((单词列表,单词个数),标记列表)

    def __generator_fn(self,words_path,tags_path):
        '读取data和tag,返回一个由两者组成的元组的生成器'
        'words_path: data的路径; tags_path: tag的路径'
        with Path(words_path).open('r',encoding='utf-8') as f_words,Path(tags_path).open('r',encoding='utf-8') as f_tags:
            for line_words,line_tags in zip(f_words,f_tags):
                yield self.__parse_fn(line_words,line_tags)

    def __input_fn(self,words_path,tags_path,params=None):
        '读取训练数据集和标记集,返回一个tf.data.Dataset对象'
        'words_path: data的路径; tags_path: tag的路径; params: 参数词典;'
        # 判断参数字典params是否为空,如果为空,赋予一个空字典,并在随后为每个参数赋予默认值
        params = params if params is not None else {}
        # tf.data.Dataset输出数据的形状:((per_line_word_list,per_line_word_num),per_line_tag_list)
        # 由于每行单词个数和标记个数不定,故用[None]表示这是个1阶的字符串张量,()表示单词个数是个0阶的标量;
        shapes = (([None],()),[None])
        # 输出的数据每个维度的类型
        types = ((tf.string,tf.int32),tf.string)
        # 默认的填充值: 我们需要将每batch的数据填充到等长，这是其默认值,注意其形状与shapes一致
        padding_value = (('<pad>',0),params.get('padding_tag','pad'))

        # 构造tf.data.Dataset
        dataset = tf.data.Dataset.from_generator(
            functools.partial(self.__generator_fn,words_path,tags_path),
            output_shapes=shapes,output_types=types
        ) # 此数据集每条数据的维度: shapes=(([None],()),[None])

        # 是否混洗以及迭代训练的次数
        # shuffle_and_repeat: 是否对数据集进行混洗和重复
        if params['shuffle_and_repeat']:
            # shapes = (batch_size,(([None],()),[None]))
            dataset = dataset.shuffle(params.get('buffer',15000)).repeat(params.get('epochs',25))

        # 将每batch数据填充到等长,便于lstm运算
        dataset = (dataset.padded_batch(params.get('batch_size',20),shapes,padding_value).prefetch(1)) # 添加预取缓冲区中元素个数
        return dataset

    def __model_fn(self,features,labels,mode,params):
        '模型结构: Bi_LSTM + CRF'
        'features: 特征列; labels: tag列; '
        'mode: tf.estimator.Estimator()自带的参数,用于判定TRAIN EVAL PREDICT三种类型'
        'params: 参数词典'
        # 判断features是那种类型:类型1:(([None],()),[None]),这是self.__input_fn()输出的类型
        # 类型2: {'words':[word1,word2,..],'nwords':number},这是我们在预测时输入的类型
        if isinstance(features,dict):
            features = features['words'],features['nwords']

        with tf.name_scope('Read_data'):
            # 获取特征列各项
            words,nwords = features # words是单词列表,nwords是其相应的数量
            # 获取词包,eg: {word1:int64}
            vocab_words = tf.contrib.lookup.index_table_from_file(params['vocabulary'],
                                                              num_oov_buckets=params['num_oov_buckets'])
            # 获取标记对应的索引，不包括用于填充batch的padding_tag
            with Path(params['tags']).open('r',encoding='utf-8') as fi:
                # indices用于存储正类tag的索引,即不包含padding_tag
                indices = [idx for idx, tag in enumerate(fi) if tag.strip() != params.get('padding_tag','pad')]
                num_tags = len(indices) + 1 # 总体的tag数量还要加上padding_tag,用于构建转移矩阵
            # 判断模式:训练，评估，预测
            training = (mode == tf.estimator.ModeKeys.TRAIN)


        with tf.name_scope('Word_Embeddings'):
            word_ids = vocab_words.lookup(words) # 获取单词列表的id列表
            # 是否加载外部的词向量
            if params['if_load_word2vec']:
                word2vec = np.load(params['word2vec'])['embeddings'] # 加载词向量,可通过word_id查找获取
                # 为padding_tag添加词向量,用全0向量表示,注意shape要保持一致
                word2vec = np.vstack([word2vec, [[0.] * params['word2vec_dim']]])
                word2vec = tf.Variable(word2vec, dtype=tf.float32, trainable=False)  # 词向量表转为tf.tensor，不可训练
                # 获取单词列表中每个单词的词向量,由于是batch，故shape= (batch_size,time_len,input_size)
                embeddings = tf.nn.embedding_lookup(word2vec, word_ids)
            else:
                # 通过模型训练词向量
                with Path(params['vocabulary']).open('r',encoding='utf-8') as fi:
                    vocab = [word for idx , word in enumerate(fi) if word.strip() != '']
                word2vec = tf.get_variable('word2vec',[len(vocab), params['word2vec_dim']])
                # 为padding_tag添加词向量,用全0向量表示,注意shape要保持一致
                padding_vec = tf.Variable([[0.]*params['word2vec_dim']],dtype=tf.float32)
                word2vec = tf.concat([word2vec,padding_vec],axis=0)
                word2vec = tf.Variable(word2vec,dtype=tf.float32,trainable=True) # 词向量表转为tf.tensor，可训练
                # 这里需要注意，padding_tag的向量应该是全0，但是在训练词向量过程中，padding_tag难以保持为全0
                # 因此需要特别处理一下，每次都需要将word2vec最后一个向量变为全0，我们用mask
                # 再构建一张lookup_table，形状与word2vec一致，其中除了最后一行元素全为0外，其余都是1
                mask = [params['word2vec_dim']]*len(vocab)+[0]*params['word2vec_dim']
                mask_lookup_table = tf.sequence_mask(mask,dtype=tf.float32)
                mask_vec = tf.nn.embedding_lookup(mask_lookup_table,word_ids)
                # 获取单词列表中每个单词的词向量,由于是batch，故shape= (batch_size,time_len,input_size)
                embeddings = tf.nn.embedding_lookup(word2vec,word_ids)
                # 将word_ids中的padding_tag的向量重置为0
                embeddings = tf.multiply(embeddings,mask_vec)

        with tf.name_scope('Embedding_Dropout'):
            embeddings = tf.layers.dropout(embeddings,rate=params['dropout'],training=training)

        with tf.name_scope('Bi_LSTM'):
            # 将输入形状转为shape=(time_len,batch_size,input_size),方便LSTM计算
            inputs = tf.transpose(embeddings,perm=[1,0,2])
            # 正向LSTM
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
            # 反向LSTM
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            # 正向每时刻隐藏层状态
            output_fw,_ = lstm_cell_fw(inputs,dtype=tf.float32,sequence_length=nwords)
            # 反向每时刻隐藏层状态
            output_bw,_ = lstm_cell_bw(inputs,dtype=tf.float32,sequence_length=nwords)
            # 将两个方向的状态，按时刻前后拼接在一起,沿最后一轴拼接
            output = tf.concat([output_fw,output_bw],axis=-1)
            # 将output形状再变回来shape = (batch_size,time_len,input_size)
            output = tf.transpose(output,perm=[1,0,2])

        with tf.name_scope('LSTM_dropout'):
            output = tf.layers.dropout(output,rate=params['dropout'],training=training)

        with tf.name_scope('Fully_connected_layer'):
            # 全连接层计算每一时刻的得分值
            logits = tf.layers.dense(output,num_tags)

        with tf.name_scope('CRF'):
            #CRF转移矩阵
            crf_params = tf.get_variable('crf',[num_tags,num_tags],dtype=tf.float32)
            # crf解码,pred_ids是预测的标记列表
            pred_ids,_ = tf.contrib.crf.crf_decode(logits,crf_params,nwords)

        # 判断是训练,评估,还是预测
        if mode == tf.estimator.ModeKeys.PREDICT:
            # 预测
            # 获取标记tag与其索引的字典,格式为{id:tag,..}
            reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
                params['tags'])
            # 将tag的id映射到tag上，获取预测的标记tag
            pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
            # 此字典存储，需要预测的内容
            predictions = {
                'pred_ids': pred_ids,
                'tags': pred_strings
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        else:
            # Loss
            # 获取标记与其索引映射表,{tag:id},注意包含了填充标记pad
            vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
            # 将真实tag转为id序列
            tags = vocab_tags.lookup(labels)
            # 计算损失函数，负的对数似然
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                logits, tags, nwords, crf_params)
            loss = tf.reduce_mean(-log_likelihood)

            # 评估指标
            weights = tf.sequence_mask(nwords)
            metrics = {
                'acc': tf.metrics.accuracy(tags, pred_ids, weights),
                'precision': precision(tags, pred_ids, num_tags, indices, weights),
                'recall': recall(tags, pred_ids, num_tags, indices, weights),
                'f1': f1(tags, pred_ids, num_tags, indices, weights),
            }

            for metric_name, op in metrics.items():
                tf.summary.scalar(metric_name, op[1])


            # 评估
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)
            # 训练
            elif mode == tf.estimator.ModeKeys.TRAIN:
                # 优化器
                train_op = tf.train.AdamOptimizer().minimize(
                    loss, global_step=tf.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)

    def train(self):
        '训练模型'
        # 模型参数
        params = {
            'word2vec_dim':self.cfg.getint('params','word2vec_dim'),# 词向量的维度
            'dropout':self.cfg.getfloat('params','dropout'),
            'num_oov_buckets':self.cfg.getint('params','num_oov_buckets'),
            'epochs':self.cfg.getint('params','epochs'),
            'batch_size':self.cfg.getint('params','batch_size'),
            'buffer':self.cfg.getint('params','buffer'),
            'lstm_size':self.cfg.getint('params','lstm_size'),
            'shuffle_and_repeat':self.cfg.getint('params','shuffle_and_repeat'),
            'if_load_word2vec':self.cfg.getint('params','if_load_word2vec'),# 是否加载自己的词向量
            'padding_tag':self.cfg.get('params','padding_tag'),# 填充符号tag
            'vocabulary':self.cfg.get('data','vocabulary'),# 词包路径
            'tags':self.cfg.get('data','tags'), # 存储tag的所有类别的路径
            'word2vec':self.cfg.get('data','word2vec'), # 词向量路径
            'train_data':self.cfg.get('data','train_data'), # 训练数据存储路径
            'train_tag':self.cfg.get('data','train_tag'), # 训练数据对应的tag的路径
            'validation_data':self.cfg.get('data','validation_data'),
            'validation_tag':self.cfg.get('data','validation_tag'),
            'test_data':self.cfg.get('data','test_data'),
            'test_tag':self.cfg.get('data','test_tag'),
            'model':self.cfg.get('model','model_path'), # 模型存储路径
            'log':self.cfg.get('log','log_path'), # 日志存储路径
            'result':self.cfg.get('result','result_path') # 预测结果存储路径
        }

        # 日志配置
        tf.logging.set_verbosity(logging.INFO)
        handlers = [
            logging.FileHandler(params['log']),
            logging.StreamHandler(sys.stdout)
        ]
        logging.getLogger('tensorflow').handlers = handlers

        # Estimator,train and evaluate
        # 训练集，评估集
        train_inpf = functools.partial(self.__input_fn,params['train_data'],params['train_tag'],
                                       params)
        eval_inpf = functools.partial(self.__input_fn,params['validation_data'],params['validation_tag'],
                                      params)
        tf_cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
        # Estimator
        estimator = tf.estimator.Estimator(self.__model_fn,params['model'],tf_cfg,params)
        Path(estimator.eval_dir()).mkdir(parents=True,exist_ok=True) # 模型存储路径如果不存在，则创建
        # early stopping，评估指标f1值,如果距离上次f1值增长500轮后,f1值未增长,则停止训练;最小训练步次8000
        hook = tf.contrib.estimator.stop_if_no_increase_hook(
            estimator,'f1',500,min_steps=8000,run_every_secs=120
        )
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])  # 训练
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)  # 评估
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        self.estimator = estimator # 保存模型

        return estimator

    def predict(self,name):
        '预测: 适用于批量预测'
        'name: 是预测的数据集类别，可取: train,validation,test'
        result_path = self.cfg.get('result','result_path') # 预测结果路径目录
        data_path = self.cfg.get('data',name+'_data') # 待预测的数据路径
        tag_path = self.cfg.get('data',name+'_tag') # 待预测的数据对应的Tag路径
        Path(result_path).mkdir(parents=True,exist_ok=True) # 判断结果路径目录是否存在，不存在则创建
        with Path(result_path+'/{}.preds.txt'.format(name)).open('wb') as fi:
            # 将待预测的数据集包装成tf.data.Dataset数据集形式
            data_inpf = functools.partial(self.__input_fn,data_path,tag_path)
            # 再读取一次待预测的数据集，用于和预测结果同时输出
            golds_gen = self.__generator_fn(data_path,tag_path)
            # 预测
            preds_gen = self.estimator.predict(data_inpf)
            for golds,preds in zip(golds_gen,preds_gen):
                ((words,_),tags) = golds
                for word,tag,tag_pred in zip(words,tags,preds['tags']):
                    fi.write(b' '.join([word,tag,tag_pred]) + b'\n')
                fi.write(b'\n')

if __name__ == '__main__':
    model = TrainModel()
    estimator = model.train() # 训练模型
    # 模型预测

    # 单条数据预测
    LINE = 'John lives in New York'
    def predict_input_fn(line):
        '格式化输入为模型需要的格式:（feature,label）'
        words = [w.encode() for w in line.split()]
        nwords = len(words)
        # 包裹成tensor
        words = tf.constant([words],dtype=tf.string)
        nwords = tf.constant([nwords],dtype=tf.int32)
        return (words,nwords),None
    predict_inpf = functools.partial(predict_input_fn, LINE)
    result = estimator.predict(predict_inpf)
    for pred in result:
        print(pred['tags'])
        break
    '''
    [b'B-PER' b'O' b'O' b'B-LOC' b'I-LOC']
    '''

    # 数据集预测
    for name in ['train','validation','test']:
        model.predict(name)