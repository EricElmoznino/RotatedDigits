import tensorflow as tf
import Helpers as hp
import numpy as np
import os
import shutil
import time

train_path = './train'

inputs_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
labels_placeholder = tf.placeholder(tf.float32, [None, 1])
keep_prob_placeholder =  tf.placeholder(tf.float32)


def loss_for_queue(sess, loss, queue, n_batches):
    error = 0
    for batch in range(n_batches):
        inputs, labels = sess.run(queue)
        error += sess.run(loss,
                          feed_dict={inputs_placeholder: inputs,
                                     labels_placeholder: labels,
                                     keep_prob_placeholder: 1.0})
    return error/n_batches


def image_input_queue(file_paths, labels, img_shape, label_shape, batch_size=50):
    file_paths_t = tf.convert_to_tensor(file_paths, dtype=tf.string)
    labels_t = tf.convert_to_tensor(labels, dtype=tf.float32)

    input_queue = tf.train.slice_input_producer(
        [file_paths_t, labels_t],
        shuffle=False)
    file_content = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(file_content, channels=img_shape[2])
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255
    label = input_queue[1]

    image.set_shape(img_shape)
    label.set_shape(label_shape)

    min_queue_examples = 256
    image_batches, label_batches = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples,
    )

    return image_batches, label_batches


def process_data(train_path, test_path, validation_path, batch_size):
    train_files = hp.path_to_files(train_path)
    train_labels = [float(file.split('/')[-1].split('_')[1]) for file in train_files]
    train_labels = np.reshape(np.asarray(train_labels), [-1, 1])
    test_files = hp.path_to_files(test_path)
    test_labels = [float(file.split('/')[-1].split('_')[1]) for file in test_files]
    test_labels = np.reshape(np.asarray(test_labels), [-1, 1])
    validation_files = hp.path_to_files(validation_path)
    validation_labels = [float(file.split('/')[-1].split('_')[1]) for file in validation_files]
    validation_labels = np.reshape(np.asarray(validation_labels), [-1, 1])

    train_queue = image_input_queue(train_files, train_labels,
                                    img_shape=[28, 28, 1], label_shape=[1],
                                    batch_size=batch_size)
    test_queue = image_input_queue(test_files, test_labels,
                                   img_shape=[28, 28, 1], label_shape=[1])
    validation_queue = image_input_queue(validation_files, validation_labels,
                                         img_shape=[28, 28, 1], label_shape=[1])

    return train_queue, test_queue, validation_queue



def build_model():
    filter_size = 12
    num_features = 25

    model = inputs_placeholder
    with tf.variable_scope('convolution_layer'):
        model = hp.convolve(model, [filter_size, filter_size], 1, num_features, pad=True)
        model = tf.nn.relu(model)

    with tf.variable_scope('fully_connected_layer'):
        model = tf.reshape(model, [-1, 28 * 28 * num_features])
        weights = hp.weight_variables([28 * 28 * num_features, 1], mean=0.0)
        model = tf.matmul(model, weights)

    model = tf.nn.dropout(model, keep_prob=keep_prob_placeholder)

    return model, tf.train.Saver(keep_checkpoint_every_n_hours=1)


def train(model, saver):
    epochs = 15
    batch_size = 50
    train_samples = 55000
    test_samples = 10000
    validation_samples = 5000

    train_set, test_set, validation_set = process_data('./train_data', './test_data', './validation_data',
                                                       batch_size)

    n_batches = int(train_samples / batch_size)
    n_steps = n_batches * epochs

    with tf.variable_scope('training'):
        sqr_dif = tf.square(model - labels_placeholder)
        mse = tf.reduce_mean(sqr_dif, name='mean_squared_error')
        angle_error = tf.reduce_mean(tf.sqrt(sqr_dif), name='mean_angle_error')
        tf.summary.scalar('angle_error', angle_error)
        optimizer = tf.train.AdamOptimizer().minimize(mse)

    summaries = tf.summary.merge_all()
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.mkdir(train_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(train_path, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        start_time = time.time()
        step = 0
        for epoch in range(1, epochs + 1):
            epoch_angle_error = 0
            for batch in range(n_batches):
                train_inputs, train_labels = sess.run(train_set)

                if step % max(int(n_steps / 1000), 1) == 0:
                    _, a, s = sess.run([optimizer, angle_error, summaries],
                                       feed_dict={inputs_placeholder: train_inputs,
                                                  labels_placeholder: train_labels,
                                                  keep_prob_placeholder: 0.75})
                    train_writer.add_summary(s, step)
                    hp.log_step(step, n_steps, start_time, a)
                else:
                    _, a = sess.run([optimizer, angle_error],
                                    feed_dict={inputs_placeholder: train_inputs,
                                               labels_placeholder: train_labels,
                                               keep_prob_placeholder: 0.75})

                epoch_angle_error += a
                step += 1

            hp.log_epoch(epoch, epochs, epoch_angle_error / n_batches)
            saver.save(sess, os.path.join(train_path, 'model.ckpt'), global_step=step)
            val_angle_error = loss_for_queue(sess, angle_error, validation_set, int(validation_samples / 50))
            hp.log_generic(val_angle_error, 'validation')

        test_angle_error = loss_for_queue(sess, angle_error, test_set, int(test_samples / 50))
        hp.log_generic(test_angle_error, 'test')
        saver.save(sess, os.path.join(train_path, 'model.ckpt'))

        coord.request_stop()
        coord.join(threads)


model, saver = build_model()
train(model, saver)
