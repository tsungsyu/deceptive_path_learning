from __future__ import absolute_import, division, print_function

import os
# from matplotlib import pylab as plt

import tensorflow as tf
# import numpy as np
# from keras import backend as K

def main():
    tf.enable_eager_execution()

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    batch_size = 128

    # train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

    # train_dataset_url = "/Users/hugh/.keras/datasets/iris_training.csv"

    train_dataset_url = "/Users/hugh/.keras/datasets/train_records.csv"

    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                               origin=train_dataset_url)

    print("Local copy of the dataset file: {}".format(train_dataset_fp))

    # column order in CSV file

    # column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    column_names =['(0-8)', '(1-8)', '(2-8)', '(3-8)', '(4-8)', '(5-8)', '(6-8)', '(7-8)', '(8-8)', '(0-7)', '(1-7)', '(2-7)', '(3-7)', '(4-7)', '(5-7)', '(6-7)', '(7-7)', '(8-7)', '(0-6)', '(1-6)', '(2-6)', '(3-6)', '(4-6)', '(5-6)', '(6-6)', '(7-6)', '(8-6)', '(0-5)', '(1-5)', '(2-5)', '(3-5)', '(4-5)', '(5-5)', '(6-5)', '(7-5)', '(8-5)', '(0-4)', '(1-4)', '(2-4)', '(3-4)', '(4-4)', '(5-4)', '(6-4)', '(7-4)', '(8-4)', '(0-3)', '(1-3)', '(2-3)', '(3-3)', '(4-3)', '(5-3)', '(6-3)', '(7-3)', '(8-3)', '(0-2)', '(1-2)', '(2-2)', '(3-2)', '(4-2)', '(5-2)', '(6-2)', '(7-2)', '(8-2)', '(0-1)', '(1-1)', '(2-1)', '(3-1)', '(4-1)', '(5-1)', '(6-1)', '(7-1)', '(8-1)', '(0-0)', '(1-0)', '(2-0)', '(3-0)', '(4-0)', '(5-0)', '(6-0)', '(7-0)', '(8-0)', 'goals']

    feature_names = column_names[:-1]
    label_name = column_names[-1]

    print("Features: {}".format(feature_names))
    print("Label: {}".format(label_name))

    class_names = ['(17)', '(77)']



    train_dataset = tf.contrib.data.make_csv_dataset(
        train_dataset_fp,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1)

    train_dataset = train_dataset.map(pack_features_vector)
    features, labels = next(iter(train_dataset))
    print(labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(81,)),  # input shape required
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(2)
    ])

    predictions = model(features)
    predictions[:5]

    tf.nn.softmax(predictions[:5])

    print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
    print("    Labels: {}".format(labels))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    global_step = tf.Variable(0)

    loss_value, grads = grad(model, features, labels)

    print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                              loss_value.numpy()))

    optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

    print("Step: {},         Loss: {}".format(global_step.numpy(),
                                              loss(model, features, labels).numpy()))

    """
    training from here
    """
    ## Note: Rerunning this cell uses the same model variables
    tfe = tf.contrib.eager

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 101

    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        # Training loop - using batches of 32
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step)

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))


    # test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

    # test_url = "/Users/hugh/.keras/datasets/iris_test.csv"

    test_url = "/Users/hugh/.keras/datasets/test_records.csv"


    test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                      origin=test_url)

    test_dataset = tf.contrib.data.make_csv_dataset(
        test_fp,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1,
        shuffle=True)

    test_dataset = test_dataset.map(pack_features_vector)

    test_accuracy = tfe.metrics.Accuracy()

    for (x, y) in test_dataset:
        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

    # tf.stack([y, prediction], axis=1)
    prediction_dataset = tf.convert_to_tensor([
        [8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 4., 4., 1., 2., 4., 2., 0., 8., 8., 3., 2., 1., 2., 3., 0., 0., 8., 8.,
         4., 2., 3., 0., 0., 0., 4., 8., 8., 3., 4., 3., 2., 4., 1., 2., 8., 8., 3., 2., 0., 3., 2., 1., 3., 8., 8., 1.,
         4., 0., 1., 1., 3., 3., 8., 8., 9., 1., 1., 2., 1., 3., 0., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
        [8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,9.,1.,1.,1.,4.,0.,0.,8.,8.,0.,0.,0.,0.,1.,4.,0.,8.,8.,0.,0.,0.,0.,0.,4.,0.,8.,8.,0.,0.,0.,0.,0.,1.,4.,8.,8.,0.,0.,0.,0.,0.,0.,0.,8.,8.,0.,0.,0.,0.,0.,0.,0.,8.,8.,9.,0.,0.,0.,0.,0.,0.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8],
    [8.,8.,8.,8.,8.,8.,8.,8.,8.,8.,9.,0.,0.,0.,0.,0.,0.,8.,8.,0.,0.,0.,0.,0.,0.,0.,8.,8.,0.,0.,0.,0.,0.,4.,0.,8.,8.,0.,0.,0.,0.,0.,1.,4.,8.,8.,0.,0.,0.,0.,0.,0.,0.,8.,8.,0.,0.,0.,0.,0.,0.,0.,8.,8.,9.,0.,0.,0.,0.,0.,0.,8.,8.,8.,8.,8.,8.,8.,8.,8.,8],
        [8., 8., 8., 8., 8., 8., 8., 8., 8.,
        8., 9., 0., 0., 0., 0., 0., 0., 8.,
        8., 0., 0., 0., 0., 0., 0., 0., 8.,
        8., 0., 0., 0., 0., 0., 0., 0., 8.,
        8., 0., 0., 1., 1., 1., 1., 4., 8.,
        8., 0., 0., 0., 0., 0., 0., 0., 8.,
        8., 0., 0., 0., 0., 0., 0., 0., 8.,
        8., 9., 0., 0., 0., 0., 0., 0., 8.,
        8., 8., 8., 8., 8., 8., 8., 8., 8],
    [8.,8.,8.,8.,8.,8.,8.,8.,8.,
8.,9.,0.,0.,0.,0.,0.,0.,8.,
8.,0.,0.,0.,0.,0.,0.,0.,8.,
8.,0.,0.,0.,0.,0.,0.,0.,8.,
8.,0.,0.,0.,0.,1.,1.,4.,8.,
8.,0.,0.,0.,0.,0.,0.,0.,8.,
8.,0.,0.,0.,0.,0.,0.,0.,8.,
8.,9.,0.,0.,0.,0.,0.,0.,8.,
8.,8.,8.,8.,8.,8.,8.,8.,8]])
    predictions = model(prediction_dataset)

    for i, logits in enumerate(predictions):
        # class_idx = tf.argmax(logits).numpy()
        p = tf.nn.softmax(logits)
        # name = class_names[class_idx]
        print("Example {} prediction: {} ({:4.1f}%), {} ({:4.1f}%)".format(i, class_names[0], 100 * p[0], class_names[1], 100 * p[1]))


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  return tf.cast(tf.stack(list(features.values()), axis=1), tf.float32), labels

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)




if __name__ == '__main__':
    main()
