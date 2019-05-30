# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 8:24
# @Author  : Weiyang
# @File    : util.py
'''
原始代码来源：

Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Utility functions

修改：
修改了原始的BLEU计算模块
修改了get_hypotheses的返回内容
'''

import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
# import numpy as np
import json
import os, re
import logging
from nltk.translate.bleu_score import sentence_bleu

logging.basicConfig(level=logging.INFO)

def calc_num_batches(total_num, batch_size):
    '''Calculates the number of batches.
    total_num: total sample number
    batch_size

    Returns
    number of batches, allowing for remainders.'''
    return total_num // batch_size + int(total_num % batch_size != 0)

def convert_idx_to_token_tensor(inputs, idx2token):
    '''Converts int32 tensor to string tensor.
    inputs: 1d int32 tensor. indices.
    idx2token: dictionary

    Returns
    1d string tensor.
    '''
    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.py_func(my_func, [inputs], tf.string)

# # def pad(x, maxlen):
# #     '''Pads x, list of sequences, and make it as a numpy array.
# #     x: list of sequences. e.g., [[2, 3, 4], [5, 6, 7, 8, 9], ...]
# #     maxlen: scalar
# #
# #     Returns
# #     numpy int32 array of (len(x), maxlen)
# #     '''
# #     padded = []
# #     for seq in x:
# #         seq += [0] * (maxlen - len(seq))
# #         padded.append(seq)
# #
# #     arry = np.array(padded, np.int32)
# #     assert arry.shape == (len(x), maxlen), "Failed to make an array"
#
#     return arry

def postprocess(hypotheses, idx2token):
    '''Processes translation outputs.
    hypotheses: list of encoded predictions
    idx2token: dictionary

    Returns
    processed hypotheses
    '''
    _hypotheses = []
    for h in hypotheses:
        sent = "".join(idx2token[idx] for idx in h)
        sent = sent.split("</s>")[0].strip()
        sent = sent.replace("▁", " ") # remove bpe symbols
        _hypotheses.append(sent.strip())
    return _hypotheses

def save_hparams(hparams, path):
    '''Saves hparams to path
    hparams: argsparse object.
    path: output directory.

    Writes
    hparams as literal dictionary to path.
    '''
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w',encoding='utf-8') as fout:
        fout.write(hp)

def load_hparams(parser, path):
    '''Loads hparams and overrides parser
    parser: argsparse parser
    path: directory or file where hparams are saved
    '''
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path, "hparams"), 'r',encoding='utf-8').read()
    flag2val = json.loads(d)
    for f, v in flag2val.items():
        parser.f = v

def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path

    Writes
    a text file named fpath.
    '''
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape

        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")

def get_hypotheses(num_batches, num_samples, sess, tensor,refs, dict):
    '''Gets hypotheses.
    num_batches: scalar.
    num_samples: scalar.
    sess: tensorflow sess object
    tensor: target tensor to fetch
    refs: reference to fetch
    dict: idx2token dictionary

    Returns
    hypotheses: list of sents
    '''
    hypotheses = []
    _refs = []
    for _ in range(num_batches):
        h ,ref= sess.run([tensor,refs])
        hypotheses.extend(h.tolist())
        _refs.extend(ref.tolist())
    hypotheses = postprocess(hypotheses, dict)

    return hypotheses[:num_samples],_refs[:num_samples]

def calc_bleu(ref,translation,name='test',epoch=0):
    '''Calculates bleu score and appends the report to translation
    ref: reference file path
    translation: model output file path
    name: result file name prefix
    epoch: epoch number

    Returns
    translation that the bleu score is appended to'''
    with open(ref,'r',encoding='utf-8') as f_ref,open(translation,'r',encoding='utf-8') as f_trans:
        ref_lst = f_ref.read().split('\n')
        trans_lst = f_trans.read().split('\n')
    # 判断参考译文与模型生成译文的数量是否一致
    assert len(ref_lst) == len(trans_lst),"参考译文和模型生成译文的数量不一致"
    fi = open("../result/"+name+"bleu.score", "a+",encoding='utf-8')
    bleus = [] # 存储每条翻译的bleu值
    for ref,trans in zip(ref_lst,trans_lst):
        ref = [list(ref)]
        trans = list(trans)
        bleu = sentence_bleu(ref,trans)
        bleus.append(bleu)
    # 求bleu的平均值
    bleu_score = sum(bleus)/float(len(bleus)) * 100
    fi.write(str(epoch)+':'+str(bleu_score)+'\n')
    fi.close()
    logging.info('Epoch: %d  BLEU Score: %.2f'% (epoch,bleu_score))

# def get_inference_variables(ckpt, filter):
#     reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
#     var_to_shape_map = reader.get_variable_to_shape_map()
#     vars = [v for v in sorted(var_to_shape_map) if filter not in v]
#     return vars