
import os
import tensorflow as tf
from model import Project_model
from processor import process_function
from text_loader import TextLoader
import numpy as np


#超参数
epochs = 10
batch_size = 16
max_len = 64
lr = 5e-6  # 学习率
keep_prob = 0.8
bert_root = './bert_model_chinese'
bert_vocab_file = os.path.join(bert_root, 'vocab.txt')
model_save_path = './model/history.model'

#获取数据
data_path = './data'
train_input,eval_input,predict_input =process_function(data_path,bert_vocab_file,True,True,True,
                                               './temp',max_len,batch_size)
def train():
    model = Project_model(bert_root,data_path,'./temp',model_save_path,batch_size,max_len,lr,keep_prob)
    with tf.Session() as sess:
        # with tf.device('/gpu:0'):
        writer = tf.summary.FileWriter('./tf_log/', sess.graph)
        # saver = tf.train.Saver()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        data_loader = TextLoader(train_input,batch_size)
        for i in range(epochs):
            data_loader.shuff()
            for j in range(data_loader.num_batches):
                x_train,y_train = data_loader.next_batch(j)
                # print(y_train.shape)
                # print(y_train)
                step, loss_= model.run_step(sess,x_train,y_train)

                print('the epoch number is : %d the index of batch is :%d, the loss value is :%f'%(i, j, loss_))


# import matplotlib.pyplot as plt
# import itertools
#
# def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
#     plt.imshow(cm, interpolation='nearest', cmap=None)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=0)
#     plt.yticks(tick_marks, classes)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
#
# def plot_matrix(y_true, y_pred):
#     from sklearn.metrics import confusion_matrix
#     confusion_matrix = confusion_matrix(y_true, y_pred)
#     class_names = ['positive', 'negative','middle']
#     plot_confusion_matrix(confusion_matrix
#                           , classes=class_names
#                           , title='Confusion matrix')
# def test():
#     data_loader = TextLoader(predict_input,batch_size)
#     saver = tf.train.import_meta_graph('./model/simi.model.meta')
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         saver.restore(sess, tf.train.latest_checkpoint('./model/'))
#         inputs_id = tf.get_default_graph().get_tensor_by_name('input_ids:0')
#         inputs_pos = tf.get_default_graph().get_tensor_by_name('input_masks:0')
#         inputs_type = tf.get_default_graph().get_tensor_by_name('segment_ids:0')
#         y = tf.get_default_graph().get_tensor_by_name('y:0')
#         true_label = []
#         pre_label = []
#         for i in range(data_loader.num_batches):
#             x_test, label = data_loader.next_batch(i)
#             x_input_ids = x_test[:, 0]
#             x_input_mask = x_test[:, 1]
#             x_segment_ids = x_test[:, 2]
#             prediction = sess.run(y,feed_dict={inputs_id: x_input_ids, inputs_pos: x_input_mask,
#                                                       inputs_type: x_segment_ids})
#             prediction = np.argmax(prediction,1)
#             for i in label:
#                 true_label.append(i)
#             for j in prediction:
#                 pre_label.append(j)
#     plot_matrix(true_label, pre_label)
#
#
#
#
# def test2():
#     data_loader = TextLoader(predict_input, batch_size)
#     g = tf.Graph()
#     # g_seesion = tf.Session(graph=g)
#     with g.as_default():
#         with tf.Session() as sess:
#         # with g.as_default():
#             sess.run(tf.global_variables_initializer())
#             with tf.gfile.GFile('./model/simi_frozen.pb', "rb") as f:
#                 graph_def = tf.GraphDef()
#                 graph_def.ParseFromString(f.read())
#             tf.import_graph_def(
#                 graph_def,
#                 input_map=None,
#                 return_elements=None,
#                 name='',
#                 op_dict=None,
#                 producer_op_list=None
#             )
#
#
#             inputs_id = tf.get_default_graph().get_tensor_by_name('input_ids:0')
#             inputs_pos = tf.get_default_graph().get_tensor_by_name('input_masks:0')
#             inputs_type = tf.get_default_graph().get_tensor_by_name('segment_ids:0')
#             y = tf.get_default_graph().get_tensor_by_name('y:0')
#             print('222222222222')
#             true_label = []
#             pre_label = []
#             for i in range(data_loader.num_batches):
#                 x_test, label = data_loader.next_batch(i)
#                 x_input_ids = x_test[:, 0]
#                 x_input_mask = x_test[:, 1]
#                 x_segment_ids = x_test[:, 2]
#                 prediction = y.eval(feed_dict={inputs_id: x_input_ids, inputs_pos: x_input_mask,
#                                                inputs_type: x_segment_ids})
#                 prediction = np.argmax(prediction, 1)
#                 print(i)
#                 for i in label:
#                     true_label.append(i)
#                 for j in prediction:
#                     pre_label.append(j)
#             print('33333333333333333')
#             print('true_label is :',true_label)
#             print('pre_label',pre_label)
#             print(len(true_label))
#             print(len(pre_label))
#             print(true_label.shape)
#             print(pre_label.shape)



if __name__ == '__main__':
    train()


