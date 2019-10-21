#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 20:28
# @Author  : song
# @File    : app.py
import os
import tensorflow as tf
from classification.data_helper import *

from model.text_cnn import TextCnn

config = {
    'n_class': 2,
    'embed_size': 128,
    'kernel_size': [3, 4, 5],
    'n_filters': 50,
    'top_k': 1,
    'lr': 1e-3
}
if __name__ == '__main__':
    neg_file = r"F:\share_ubuntu\my_prj\data\nlp\classification/neg.xls"
    pos_file = r"F:\share_ubuntu\my_prj\data\nlp\classification/pos.xls"
    data = prepare_data(neg_file, pos_file)
    data['comm'] = data['comm'].apply(clean_content)
    word_id, id_word = build_vocab(list(data['comm']))
    print(word_id)
    train_frac = 0.7
    train_size = int(train_frac * data.shape[0])

    train_x, test_x = list(data['comm'][:train_size]), list(data['comm'][train_size:])
    train_y, test_y = list(data['label'][:train_size]), list(data['label'][train_size:])
    config['vocab_size'] = len(word_id)

    summary_train_dir = "./output/summary/train"
    summary_dev_dir = "./output/summary/dev"
    ckpt_dir = "./output/ckpt/txt_clf"

    writer_summary_train = tf.summary.FileWriter(summary_train_dir)
    writer_summary_dev = tf.summary.FileWriter(summary_dev_dir)

    epoch = 14000
    with tf.Session() as sess:
        model = TextCnn(**config)
        cpkt = tf.train.get_checkpoint_state(ckpt_dir)
        if cpkt and cpkt.model_checkpoint_path:
            model.saver.restore(sess, cpkt.model_checkpoint_path)
        writer_summary_train.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        f1_max = 0
        f1_count = 0
        for i in range(epoch):
            for item in generate_batch(train_x, train_y, word_id=word_id):
                global_step, loss_train, acc_train, summary_train = model.train(sess, batch_x=item[0], batch_y=item[1])

                writer_summary_train.add_summary(summary_train, global_step)

                if global_step % 100 == 0:
                    batch_x, batch_y = get_data(test_x, test_y, word_id=word_id)
                    acc_test, f1, loss_test, global_step, summary_dev = model.evaluate(sess, batch_x=batch_x,
                                                                                       batch_y=batch_y)

                    writer_summary_dev.add_summary(summary_dev, global_step)
                    writer_summary_dev.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag='eval_f1', simple_value=f1)]), global_step)
                    print(
                        'step:{}\tloss_train:{:.4f} acc_train:{:.4f} loss_test:{:.4f} acc_test:{:.4f} f1_test:{:.4f}'.format(
                            global_step,
                            loss_train, acc_train, loss_test, acc_test, f1))
                    print("**********************f1_max:", f1_max)

                    if f1 > f1_max:
                        f1_max = f1
                        f1_count = 0
                        model.saver.save(sess, ckpt_dir, global_step=global_step)
                    else:
                        f1_count += 1
                    if f1_count >= 5:
                        print('early stop,step:%s' % global_step)
                        exit(0)
