#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   add_op.py
@Contact :   384474737@qq.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
19-4-1 下午1:37   alpha      1.0         None
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
from nets import inception_v3
from nets import nets_factory
from tensorflow.python.framework import graph_util


def freeze_model_from_forward_ckpt(ckpt_dir, output_node_name, model_path):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    ckpt_path = ckpt.model_checkpoint_path
    # output_node_name = 'InceptionV3/Logits/classes,output'
    network_fn = nets_factory.get_network_fn('inception_v3', 5)

    x = tf.placeholder(shape=[None, None, None, 3], dtype=tf.uint8, name="input")
    logits, end_points = network_fn(x)
    saver = tf.train.Saver()
    print("freeze model from ckpt :", ckpt_path)
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_name.split(",")  # 如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(model_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("finish frozen model!")
