# -*- coding: utf-8 -*-
# @Time    : 2019/5/9 8:24
# @Author  : Weiyang
# @File    : ModelPredict.py
'''
加载训练好的模型预测: 用于输入一条语句，返回一个结果序列
'''
from pathlib import Path
import tensorflow as tf
import functools
import numpy as np
from tf_metrics import precision,recall,f1

class ModelPredict(object):
    '加载训练好的模型，用于预测'

    def __init__(self):
        self.cfg = self.__getConfig()  # configparser.ConfigParser().read()返回的对象
        self.params = self.__getParmas() # 存储模型参数
        self.model = self.__loadModel() # 存储训练好的模型

    def __getConfig(self):
        '读取配置'
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read('../config/config.ini',encoding='utf-8')
        return cfg

    def __getParmas(self):
        '加载模型参数'
        # 模型参数
        params = {
            'char2vec_dim': self.cfg.getint('params', 'char2vec_dim'),  # 汉语单字或英文字母向量的维度
            'word2vec_dim': self.cfg.getint('params', 'word2vec_dim'),  # 汉语词汇或英文单词的词向量的维度
            'dropout': self.cfg.getfloat('params', 'dropout'),
            'num_oov_buckets': self.cfg.getint('params', 'num_oov_buckets'),
            'epochs': self.cfg.getint('params', 'epochs'),
            'batch_size': self.cfg.getint('params', 'batch_size'),
            'buffer': self.cfg.getint('params', 'buffer'),
            'char_lstm_size': self.cfg.getint('params', 'char_lstm_size'),  # 汉语单字或英文字母的LSTM的维度
            'word_lstm_size': self.cfg.getint('params', 'word_lstm_size'),  # 汉语词语或英文单词的LSTM的维度
            'shuffle_and_repeat': self.cfg.getint('params', 'shuffle_and_repeat'),
            'if_load_char2vec': self.cfg.getint('params', 'if_load_char2vec'),  # 是否加载自己的汉语单字或英文字母的向量
            'if_load_word2vec': self.cfg.getint('params', 'if_load_word2vec'),  # 是否加载自己的汉语词语或英文单词的向量
            'padding_tag': self.cfg.get('params', 'padding_tag'),  # 填充符号tag
            'char_vocabulary': self.cfg.get('data', 'char_vocabulary'),  # 汉语单字或英文字母的词包路径
            'word_vocabulary': self.cfg.get('data', 'word_vocabulary'),  # 汉语词汇或英文单词的词包路径
            'tags': self.cfg.get('data', 'tags'),  # 存储tag的所有类别的路径
            'char2vec': self.cfg.get('data', 'char2vec'),  # 汉语单字或英文字母的词向量路径
            'word2vec': self.cfg.get('data', 'word2vec'),  # 汉语词汇或英文单词的词向量路径
            'train_data': self.cfg.get('data', 'train_data'),  # 训练数据存储路径
            'train_tag': self.cfg.get('data', 'train_tag'),  # 训练数据对应的tag的路径
            'validation_data': self.cfg.get('data', 'validation_data'),
            'validation_tag': self.cfg.get('data', 'validation_tag'),
            'test_data': self.cfg.get('data', 'test_data'),
            'test_tag': self.cfg.get('data', 'test_tag'),
            'model': self.cfg.get('model', 'model_path'),  # 模型存储路径
            'log': self.cfg.get('log', 'log_path'),  # 日志存储路径
            'result': self.cfg.get('result', 'result_path')  # 预测结果存储路径
        }
        return params

    def __model_fn(self,features,labels,mode,params):
        '模型结构: Bi_LSTM + CRF'
        'features: 特征列; labels: tag列; '
        'mode: tf.estimator.Estimator()自带的参数,用于判定TRAIN EVAL PREDICT三种类型'
        'params: 参数词典'
        # 判断features是那种类型:类型1:((([None],()),([None,None],[None])),[None]),这是self.__input_fn()输出的类型
        # 类型2: {'words':[word1,word2,..],'nwords':number,'chars':[['J','o',..],['l',..],..],'nchars':number},
        # 这是我们在预测时输入的类型
        if isinstance(features, dict):
            features = ((features['words'], features['nwords']),
                        (features['chars'], features['nchars']))

        with tf.name_scope('Read_data'):
            # 获取特征列各项
            (words, nwords), (chars, nchars) = features  # words是单词列表,nwords是其相应的数量
            # 获取汉语单字或英文字母的词包,eg: {char1:int64}
            vocab_chars = tf.contrib.lookup.index_table_from_file(params['char_vocabulary'],
                                                                  num_oov_buckets=params['num_oov_buckets'])
            # 获取汉语词语或英文单词的词包,eg:{char2:int64}
            vocab_words = tf.contrib.lookup.index_table_from_file(params['word_vocabulary'],
                                                                  num_oov_buckets=params['num_oov_buckets'])
            # 获取标记对应的索引，不包括用于填充batch的padding_tag
            with Path(params['tags']).open('r', encoding='utf-8') as fi:
                # indices用于存储正类tag的索引,即不包含padding_tag
                indices = [idx for idx, tag in enumerate(fi) if tag.strip() != params.get('padding_tag', 'pad')]
                num_tags = len(indices) + 1  # 总体的tag数量还要加上padding_tag,用于构建转移矩阵
            # 获取汉语单字或英文字母的数量
            with Path(params['char_vocabulary']).open('r', encoding='utf-8') as fi:
                # # char的数量还得加上,不在词包中的字符我们给它们的索引数量
                num_chars = sum(1 for _ in fi) + params['num_oov_buckets']
            # 判断模式:训练，评估，预测
            training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.name_scope('Char_Embeddings_Layer'):
            char_ids = vocab_chars.lookup(chars)  # 获取字母列表的id列表
            # char2vec = tf.get_variable('char_embeddings',[num_chars,params['char2vec_dim']],tf.float32)
            # char_embeddings = tf.nn.embedding_lookup(char2vec,char_ids)
            # 是否加载外部的汉字单字或英文字母的向量
            if params['if_load_char2vec']:
                char2vec = np.load(params['char2vec'])['embeddings']  # 加载词向量,可通过char_id查找获取
                # 为padding_tag添加词向量,用全0向量表示,注意shape要保持一致
                char2vec = np.vstack([char2vec, [[0.] * params['char2vec_dim']]])
                char2vec = tf.Variable(char2vec, dtype=tf.float32, trainable=False)  # 词向量表转为tf.tensor，不可训练
                # 获取字母列表中每个字母的词向量,由于是batch，故shape= (batch_size,time_len,input_size)
                # 这里batch是每条输入中的单词个数
                char_embeddings = tf.nn.embedding_lookup(char2vec, char_ids)
            else:
                # 通过模型训练词向量
                with Path(params['char_vocabulary']).open('r', encoding='utf-8') as fi:
                    char_vocab = [word for idx, word in enumerate(fi) if word.strip() != '']
                char2vec = tf.get_variable('char2vec', [len(char_vocab), params['char2vec_dim']])
                # 为padding_tag添加词向量,用全0向量表示,注意shape要保持一致
                padding_vec = tf.Variable([[0.] * params['char2vec_dim']], dtype=tf.float32)
                char2vec = tf.concat([char2vec, padding_vec], axis=0)
                char2vec = tf.Variable(char2vec, dtype=tf.float32, trainable=True)  # 词向量表转为tf.tensor，可训练
                # 这里需要注意，padding_tag的向量应该是全0，但是在训练词向量过程中，padding_tag难以保持为全0
                # 因此需要特别处理一下，每次都需要将char2vec最后一个向量变为全0，我们用mask
                # 再构建一张lookup_table，形状与char2vec一致，其中除了最后一行元素全为0外，其余都是1
                mask = [params['char2vec_dim']] * len(char_vocab) + [0] * params['char2vec_dim']
                mask_lookup_table = tf.sequence_mask(mask, dtype=tf.float32)
                mask_vec = tf.nn.embedding_lookup(mask_lookup_table, char_ids)
                # 获取单词中每个字母的词向量,由于是batch，故shape= (batch_size,time_len,input_size)
                # 这里batch是每条输入中的单词个数
                embeddings = tf.nn.embedding_lookup(char2vec, char_ids)
                # 将char_ids中的padding_tag的向量重置为0
                char_embeddings = tf.multiply(embeddings, mask_vec)

        with tf.name_scope('Char_Embedding_Dropout_Layer'):
            # char_embeddings.shape = (None,None,None,params['char2vec_dim']
            # 第一个None是batch_size,第二个是每条输入中的单词个数
            # 第三个None是每条输入中每个单词包含的字母个数的列表
            char_embeddings = tf.layers.dropout(char_embeddings, rate=params['dropout'], training=training)

        with tf.name_scope('Char_LSTM_Layer'):
            dim_words = tf.shape(char_embeddings)[1]  # 当前输入中的单词个数
            dim_chars = tf.shape(char_embeddings)[2]  # 当前输入中的每个单词的字母个数
            flat = tf.reshape(char_embeddings, [-1, dim_chars, params['char2vec_dim']])
            t = tf.transpose(flat, perm=[1, 0, 2])
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['char_lstm_size'])
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_fw)
            # 获取正向LSTM最后一时刻的输出
            _, (_, output_fw) = lstm_cell_fw(t, dtype=tf.float32,
                                             sequence_length=tf.reshape(nchars, [-1]))
            # 获取反向LSTM最后一时刻的输出
            _, (_, output_bw) = lstm_cell_bw(t, dtype=tf.float32,
                                             sequence_length=tf.reshape(nchars, [-1]))
            # 将这两个时刻的输出按最后一维度拼接
            output = tf.concat([output_fw, output_bw], axis=-1)
            char_embeddings = tf.reshape(output, [-1, dim_words, params['char_lstm_size'] * 2])

        with tf.name_scope('Word_Embeddings_Layer'):
            word_ids = vocab_words.lookup(words)  # 获取单词列表的id列表
            # 是否加载外部的词向量
            if params['if_load_word2vec']:
                word2vec = np.load(params['word2vec'])['embeddings']  # 加载词向量,可通过word_id查找获取
                # 为padding_tag添加词向量,用全0向量表示,注意shape要保持一致
                word2vec = np.vstack([word2vec, [[0.] * params['word2vec_dim']]])
                word2vec = tf.Variable(word2vec, dtype=tf.float32, trainable=False)  # 词向量表转为tf.tensor，不可训练
                # 获取单词列表中每个单词的词向量,由于是batch，故shape= (batch_size,time_len,input_size)
                word_embeddings = tf.nn.embedding_lookup(word2vec, word_ids)
            else:
                # 通过模型训练词向量
                with Path(params['word_vocabulary']).open('r', encoding='utf-8') as fi:
                    vocab = [word for idx, word in enumerate(fi) if word.strip() != '']
                word2vec = tf.get_variable('word2vec', [len(vocab), params['word2vec_dim']])
                # 为padding_tag添加词向量,用全0向量表示,注意shape要保持一致
                padding_vec = tf.Variable([[0.] * params['word2vec_dim']], dtype=tf.float32)
                word2vec = tf.concat([word2vec, padding_vec], axis=0)
                word2vec = tf.Variable(word2vec, dtype=tf.float32, trainable=True)  # 词向量表转为tf.tensor，可训练
                # 这里需要注意，padding_tag的向量应该是全0，但是在训练词向量过程中，padding_tag难以保持为全0
                # 因此需要特别处理一下，每次都需要将word2vec最后一个向量变为全0，我们用mask
                # 再构建一张lookup_table，形状与word2vec一致，其中除了最后一行元素全为0外，其余都是1
                mask = [params['word2vec_dim']] * len(vocab) + [0] * params['word2vec_dim']
                mask_lookup_table = tf.sequence_mask(mask, dtype=tf.float32)
                mask_vec = tf.nn.embedding_lookup(mask_lookup_table, word_ids)
                # 获取单词列表中每个单词的词向量,由于是batch，故shape= (batch_size,time_len,input_size)
                embeddings = tf.nn.embedding_lookup(word2vec, word_ids)
                # 将word_ids中的padding_tag的向量重置为0
                word_embeddings = tf.multiply(embeddings, mask_vec)

        with tf.name_scope('Concatenate_CharEmbedding_WordEmbedding'):
            embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)

        with tf.name_scope('Dropout_Layer'):
            embeddings = tf.layers.dropout(embeddings, rate=params['dropout'], training=training)

        with tf.name_scope('Word_Bi_LSTM'):
            # 将输入形状转为shape=(time_len,batch_size,input_size),方便LSTM计算
            inputs = tf.transpose(embeddings, perm=[1, 0, 2])
            # 正向LSTM
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['word_lstm_size'])
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['word_lstm_size'])
            # 反向LSTM
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            # 正向每时刻隐藏层状态
            output_fw, _ = lstm_cell_fw(inputs, dtype=tf.float32, sequence_length=nwords)
            # 反向每时刻隐藏层状态
            output_bw, _ = lstm_cell_bw(inputs, dtype=tf.float32, sequence_length=nwords)
            # 将两个方向的状态，按时刻前后拼接在一起,沿最后一轴拼接
            output = tf.concat([output_fw, output_bw], axis=-1)
            # 将output形状再变回来shape = (batch_size,time_len,input_size)
            output = tf.transpose(output, perm=[1, 0, 2])

        with tf.name_scope('LSTM_dropout'):
            output = tf.layers.dropout(output, rate=params['dropout'], training=training)

        with tf.name_scope('Fully_connected_layer'):
            # 全连接层计算每一时刻的得分值
            logits = tf.layers.dense(output, num_tags)

        with tf.name_scope('CRF'):
            # CRF转移矩阵
            crf_params = tf.get_variable('crf', [num_tags, num_tags], dtype=tf.float32)
            # crf解码,pred_ids是预测的标记列表
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

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

    def __loadModel(self):
        '加载模型'
        model_path = self.cfg.get('model','model_path') # 模型存储路径
        estimator = tf.estimator.Estimator(self.__model_fn,model_path,params=self.params)
        return estimator

    def __predict_input_fn(self,line):
        '格式化输入为模型需要的格式:（feature,label）'
        words = [w.encode() for w in line.split()]
        nwords = len(words)
        # 包裹成tensor
        words = tf.constant([words],dtype=tf.string)
        nwords = tf.constant([nwords],dtype=tf.int32)
        # char
        chars = [[c.encode() for c in w] for w in line.strip().split()]
        nchars = [len(c) for c in chars]
        # 将每个单词扩充为等长,用padding_tag填充
        max_len = max(nchars)
        chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, nchars)]
        # 包裹成tensor
        chars = tf.constant([chars], dtype=tf.string)
        nchars = tf.constant([nchars], dtype=tf.int32)
        return ((words, nwords), (chars, nchars)), None

    def __pretty_print(self,line,preds):
        '格式化预测结果'
        words = line.strip().split()
        lengths = [max(len(w),len(p)) for w,p in zip(words,preds)]
        padded_words = [w + (l-len(w))*' ' for w,l in zip(words,lengths)]
        padded_preds = [p.decode()+(l-len(p))*' ' for p,l in zip(preds,lengths)]
        print('words: {}'.format(' '.join(padded_words)))
        print('preds: {}'.format(' '.join(padded_preds)))

    def predict(self,line):
        '模型预测'
        predict_inpf = functools.partial(self.__predict_input_fn,line)
        pred_result = self.model.predict(predict_inpf)
        for pred in pred_result:
            self.__pretty_print(line,pred['tags'])
            break

if __name__ == '__main__':
    mp = ModelPredict()
    LINE = 'John lives in New York'
    mp.predict(LINE)
    '''
    words: John  lives in New   York 
    preds: B-PER O     O  B-LOC I-LOC
    '''