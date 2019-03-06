import numpy as np, os, sys, logging, tensorflow as tf, matplotlib.pyplot as plt
from reward_function import Reward_Function
from read_record import get_iterator
import time

np.set_printoptions(suppress=True)

from constants import *
from subprocess import *
import sys

def softmax_cross_entropy(labels=None, logits=None):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits))

def show_various_examples():
    sess = tf.Session()
    n_examples_to_show = 20

    filename, iterator = get_iterator(batch_size=1,
                                      shuffle=True,
                                      buffer_size=1024)
    example = iterator.get_next()

    # Get example tensors from dataset
    X, Y = example['state'], example['label']
    Y = tf.reshape(Y, (-1,n_classes))
    # Infer class from model
    model = Reward_Function(X)
    logits = model.logits

    # Initializing model/optimizer weights
    sess.run(tf.global_variables_initializer())
    # Initialize local variables like mean on loss and accuracy
    sess.run(tf.local_variables_initializer())
    last_epoch = 0
    
    # For saving model parameters (weights)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=reward_fn_scope), 
        max_to_keep=100)
    if not os.path.exists(checkpoints_dir):
        raise ValueError('Model does not have any checkpoints saved to initialize from.  Set recover to False.')
    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    print('Restoring model from: ', latest_checkpoint)
    saver.restore(sess, latest_checkpoint)


    # Initialize the dataset iterator
    sess.run(iterator.initializer, feed_dict={filename:[os.path.join(TFRECORDS_DIRECTORY, 'val.tfrecord')]})
    for i in range(n_examples_to_show):
        ex,label,logs = sess.run([X,Y,logits])
        lab = np.argmax(label)
        pred = np.argmax(logs)
        if lab in [0,2,3]:
            print(logs)
            fig,ax_fig = plt.subplots(nrows=1,ncols=1)
            status = "Label: %d, Predicted: %d"%(lab,pred)
            img = ax_fig.imshow(ex[0],vmin=0,vmax=40); ax_fig.set_title(status)
            cb = fig.colorbar(img,ax=ax_fig,ticks=[0,10,20,30,40])
            plt.show()

if __name__ == "__main__":
    tf.reset_default_graph()
    if int(sys.argv[1]) == 2:
        show_various_examples(); exit(0)

    evaluate = int(sys.argv[1])
    if evaluate:
        batch_size = 5000

    # Hyper-parameters, constants, etc...

    recover_model = False

    sess = tf.Session()
    # Keep track of number of iterations/model_config['epochs
    global_step = tf.get_variable('global_step',shape=[],dtype=tf.int32,
        initializer=tf.zeros_initializer,trainable = False)
    global_epoch = tf.get_variable('global_epoch',shape=[],dtype=tf.int32,
        initializer=tf.zeros_initializer,trainable = False)

    filename, iterator = get_iterator(batch_size=batch_size,
                                      shuffle=True,
                                      buffer_size=1024)
    example = iterator.get_next()

    # Get example tensors from dataset
    X, Y = example['state'], example['label']
    x_shape = tf.shape(X)
    X = X + tf.random_uniform(x_shape, minval=-5,maxval=5)
    Y = tf.reshape(Y, (-1,n_classes))

    # Infer class from model
    model = Reward_Function(X)
    logits = model.logits
    # Compute loss
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
    loss = softmax_cross_entropy(labels=Y,logits=logits)
    # Get optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = optimizer.apply_gradients(zip(gradients, variables))
    #opt = optimizer.minimize(loss)
    # Keep track of loss over a period of time
    metric_loss, update_loss = tf.metrics.mean(xent, name='metrics')

    labels = tf.argmax(input=Y, axis=1)
    predictions = tf.argmax(input=logits, axis=1)
    conf_mat = tf.confusion_matrix(labels,predictions,num_classes=n_classes)
    metric_accuracy, update_accuracy = tf.metrics.accuracy(labels=labels,
                                                           predictions=predictions,
                                                           name='metrics')
    # Collect all metric-related variables
    metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics')
    reset_metrics = tf.variables_initializer(var_list=metric_vars)
    # Get TensorBoard summaries
    loss_summary = tf.summary.scalar('Loss', update_loss)
    accuracy_summary = tf.summary.scalar('Accuracy', update_accuracy)
    merged_summary = tf.summary.merge_all()
    # Initializing model/optimizer weights
    sess.run(tf.global_variables_initializer())
    # Initialize local variables like mean on loss and accuracy
    sess.run(tf.local_variables_initializer())
    last_epoch = 0
    
    # For saving model parameters (weights)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=reward_fn_scope), 
        max_to_keep=100)

    # For writing summaries to TensorBoard
    if recover_model or evaluate:
        if not os.path.exists(checkpoints_dir):
            raise ValueError('Model does not have any checkpoints saved to initialize from.  Set recover to False.')
        latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
        print('Restoring model from: ', latest_checkpoint)
        saver.restore(sess, latest_checkpoint)
        last_epoch = sess.run(global_epoch)
        #print(last_epoch); exit(0)
    else:
        call("rm -rf %s"%summaries_dir, shell=True)

    writers = [tf.summary.FileWriter(os.path.join(summaries_dir, split), sess.graph)
               for split in ['train', 'val']]
    tensorboard_train_writer, tensorboard_val_writer = writers

    if evaluate:
        # Initialize the dataset iterator
        sess.run(iterator.initializer, feed_dict={filename:[os.path.join(TFRECORDS_DIRECTORY, 'val.tfrecord')]})
        cf = sess.run([conf_mat])
        print(cf)
        print("Accuracy: %.3f"%(np.trace(cf[0])/batch_size))
        exit(0)

    for epoch in range(last_epoch, n_epochs):
        # Print current information
        print('Global Epoch: ', sess.run(global_epoch))
        print('Global Step: ', sess.run(global_step))
        print('Training Epoch ', epoch)

        # Save the model weights
        checkpoint_file_name = os.path.join(checkpoints_dir, 'checkpoint-' + str(epoch) + '.ckpt')
        print('Saving model to: ', checkpoint_file_name)
        saver.save(sess, checkpoint_file_name)

        # Initialize the dataset iterator
        sess.run(iterator.initializer, feed_dict={filename:[os.path.join(TFRECORDS_DIRECTORY, 'train.tfrecord')]})

        ### TRAINING LOOP ###
        sess.run(reset_metrics)
        n = 0
        while True:
            try:
                if (n % 1000) == 0:
                    saver.save(sess, checkpoint_file_name)
                # Run a single iteration of optimization
                l, acc, update, summary, xent_py = \
                    sess.run([loss, update_accuracy, opt, merged_summary, xent],
                              feed_dict=model.train_feed_dict)
                if n == 0:
                    print('Epoch %d train loss: %f' % (epoch, l))
                if n % 100 == 0:
                    print('Global Step',n,'Running Average Loss: ',l)
                n = n + 1
            except tf.errors.OutOfRangeError:
                # Write summary to tensorboard when epoch is finished
                tensorboard_train_writer.add_summary(summary, epoch)
                break

        # Reset the metrics for validation
        sess.run(reset_metrics)
        # Reset the dataset iterator for validation
        sess.run(iterator.initializer, feed_dict={filename:[os.path.join(TFRECORDS_DIRECTORY, 'val.tfrecord')]})
        ### VALIDATION LOOP ###
        l=None
        while True:
            try:
                _, l,accuracy, summary = \
                    sess.run([xent, loss,update_accuracy, merged_summary],
                              feed_dict=model.valid_feed_dict)
            except tf.errors.OutOfRangeError:
                print('Epoch %d validation loss: %f' % (epoch, l))
                tensorboard_val_writer.add_summary(summary, epoch)
                break

        sess.run(global_epoch.assign(global_epoch+1))

    sess.close()