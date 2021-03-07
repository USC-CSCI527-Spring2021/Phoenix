import tensorflow as tf


def read_tfrecord(serialized_example):
    discard_feature_spec = {
        "features": tf.io.FixedLenFeature((13, 34, 1), tf.int64),
        "labels": tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(serialized_example, discard_feature_spec)

    feature0 = example['features']
    feature1 = example['labels']

    return feature0, feature1
