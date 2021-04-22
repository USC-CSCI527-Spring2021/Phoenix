import argparse
import glob
import os

import dill as pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from extract_features.FeatureGenerator import FeatureGenerator
from logs_parser import chi_pon_kan_model, discarded_model_dataset
from trainer.utils import TRAIN_SPLIT, BATCH_SIZE, RANDOM_SEED

tf.random.set_seed(RANDOM_SEED)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def encoding(feature_label_tuple):
    """
      Convert to tf-example
    """
    feature, label = feature_label_tuple
    return tf.train.Example(features=tf.train.Features(
        feature={'features': _float_feature(feature.ravel().tolist()),
                 'labels': _float_feature(label.ravel().tolist())
                 }))


class PreprocessData(object):
    """
    Serialized feature specs, output file path, so we can load it when training
    """

    def __init__(
            self,
            total,
            classes_distribution,
            input_feature_spec):
        self.classes_distribution = classes_distribution
        self.total = total

        self.input_feature_spec = input_feature_spec


# a = open("tester.json", 'a')
fg = FeatureGenerator()


class Pipeline:
    def __init__(self, job_type):
        self.num_counter = 0
        self.job_type = job_type
        self.counter = 0
        self.write_count = [0] * 34 if self.job_type == "discard" else [0, 0]
        self.classes_distribution = [0] * 34 if self.job_type == "discard" else [0, 0]
        # self.train = []
        # self.eval = []
        if self.job_type == "discard":
            self.dataset = {i: [] for i in range(34)}
        else:
            self.dataset = {0: [], 1: []}
        self.log_count = 0
        self.params = {
            "discard": {
                "table_bq_table": 'mahjong.discard',
                "feature_spec": {
                    "features": tf.io.FixedLenFeature((62, 34, 1), tf.float32),
                    "labels": tf.io.FixedLenFeature((34,), tf.float32),
                },
                "process_fn": discarded_model_dataset.DiscardedFeatureExtractor(),
                "transform_fn": fg.DiscardFeatureGenerator,
            },
            "chi": {
                "table_bq_table": "mahjong.chi",
                "feature_spec": {
                    "features": tf.io.FixedLenFeature((63, 34, 1), tf.float32),
                    "labels": tf.io.FixedLenFeature((2,), tf.float32),
                },
                "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
                "transform_fn": fg.ChiFeatureGenerator,
            },
            "pon": {
                "table_bq_table": "mahjong.pon",
                "feature_spec": {
                    "features": tf.io.FixedLenFeature((63, 34, 1), tf.float32),
                    "labels": tf.io.FixedLenFeature((2,), tf.float32),
                },
                "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
                "transform_fn": fg.PonFeatureGenerator,
            },
            "kan": {
                "table_bq_table": "mahjong.kan",
                "feature_spec": {
                    "features": tf.io.FixedLenFeature((66, 34, 1), tf.float32),
                    "labels": tf.io.FixedLenFeature((2,), tf.float32),
                },
                "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
                "transform_fn": fg.KanFeatureGenerator,
            },
            "riichi": {
                "table_bq_table": "mahjong.riichi",
                "feature_spec": {
                    "features": tf.io.FixedLenFeature((62, 34, 1), tf.float32),
                    "labels": tf.io.FixedLenFeature((2,), tf.float32),
                },
                "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
                "transform_fn": fg.RiichiFeatureGenerator,
            }
        }

    def write_tfrecords(self, path_prefix, types=None, tfdata=None):
        # case 1: write multiple labels separately
        if types is not None and not tfdata:
            out_path = "{}/{}-dataset-{}".format(path_prefix, types, self.write_count[types])
            writer = tf.data.experimental.TFRecordWriter(out_path)
            writer.write(
                tf.data.Dataset.from_tensor_slices(self.dataset[types]))
        # case 2: write train, test or val dataset directly with sharding
        if tfdata and types:
            c = 0
            self.num_counter = 0
            while True:
                tmp = tfdata.take(40000)
                length = len(list(tmp))
                if length < 40000:
                    self.num_counter += length
                    out_path = "{}/{}-dataset-{}".format(path_prefix, types, c + 1)
                    writer = tf.data.experimental.TFRecordWriter(out_path)
                    writer.write(tmp)
                    if types != 'train':
                        print(f"Total {types} instance:", self.num_counter)
                    break
                out_path = "{}/{}-dataset-{}".format(path_prefix, types, c)
                writer = tf.data.experimental.TFRecordWriter(out_path)
                writer.write(tmp)
                tfdata = tfdata.skip(40000)
                self.num_counter += 40000
                c += 1


    def feature_writer(self, dataset_prefix, data):
        try:
            data_tuple = next(data)
            if data_tuple:
                _, label = data_tuple
                l = np.argmax(label)
                self.classes_distribution[int(l)] += 1

                encode_str = encoding(data_tuple).SerializeToString()
                self.counter += 1
                self.dataset[int(l)].append(encode_str)

                if len(self.dataset[int(l)]) % 40000 == 0:
                    self.write_tfrecords(dataset_prefix, int(l))
                    self.dataset[int(l)] = []
                    self.write_count[int(l)] += 1
        except StopIteration:
            pass

    def process(self, job_dir):
        dataset_prefix = os.path.join(job_dir, 'processed_data', self.job_type)
        csv_path = glob.glob(os.path.join(job_dir, "dataset/2021.csv"))
        # delete existing processed data
        if tf.io.gfile.exists(dataset_prefix):
            tf.io.gfile.rmtree(dataset_prefix)
        tf.io.gfile.makedirs(dataset_prefix)

        a = open(f"{self.job_type}_error.json", "a")
        b = open(f"{self.job_type}_error.log", "a")
        for path in csv_path:
            print(path, "Start")
            df = pd.read_csv(path)
            logs_col = df["log_content"]
            for log_str in logs_col:
                self.log_count += 1
                # if self.log_count < 4:
                #     continue
                extractors = self.params[self.job_type]["process_fn"].process(log_str)
                try:
                    # import json
                    # a.write(json.dumps(list(extractors))+'\n')
                    # if self.log_count == 50:
                    #     raise
                    if next(extractors):
                        try:
                            for data in extractors:
                                self.feature_writer(
                                    dataset_prefix, self.params[self.job_type]["transform_fn"](data))
                        except:
                            import json
                            a.write(json.dumps(list(extractors)) + '\n')
                            b.write(log_str)
                except StopIteration:
                    pass
                print("log #:", self.log_count)
            print(path, "Finished")
        a.close()
        b.close()
        for cls in range(len(self.classes_distribution)):
            if not self.dataset[cls]:
                continue
            self.write_tfrecords(dataset_prefix, cls)
        meta_data = PreprocessData(self.counter, self.classes_distribution,
                                   self.params[self.job_type]["feature_spec"])
        with tf.io.gfile.GFile(os.path.join(dataset_prefix, self.job_type + "_meta"), 'wb') as f:
            pickle.dump(meta_data, f)
        print("Total number of instance:", self.counter)
        print("Classes distribution:", self.classes_distribution)

        oversampling_prefix = os.path.join(job_dir, 'with_oversampling_data', self.job_type)
        if tf.io.gfile.exists(oversampling_prefix):
            tf.io.gfile.rmtree(oversampling_prefix)
        tf.io.gfile.makedirs(oversampling_prefix)

        train_dataset = []
        test_dataset = []
        val_dataset = []
        train_total = 0
        for i, class_dist in enumerate(self.classes_distribution):
            tfrecords = glob.glob(os.path.join("./processed_data", self.job_type, f"{i}-*"))
            dataset = tf.data.TFRecordDataset(tfrecords)
            t_size = int(class_dist * TRAIN_SPLIT)
            v_size = int(class_dist * ((1 - TRAIN_SPLIT) / 2))
            train_dataset.append(dataset.take(t_size))
            train_total += t_size
            dataset = dataset.skip(t_size)
            test_dataset.append(dataset.take(v_size))
            val_dataset.append(dataset.skip(v_size))
        tmp_test = test_dataset[0]
        for d in test_dataset[1:]:
            tmp_test.concatenate(d)
        test_dataset = tmp_test.shuffle(BATCH_SIZE)
        tmp_val = val_dataset[0]
        for d in val_dataset[1:]:
            tmp_val.concatenate(d)
        val_dataset = tmp_val.shuffle(BATCH_SIZE)
        print("Total train instance before resampling:", train_total)
        self.write_tfrecords(oversampling_prefix, "val", val_dataset)
        self.write_tfrecords(oversampling_prefix, "test", test_dataset)
        resampled_ds = tf.data.experimental \
            .sample_from_datasets(train_dataset,
                                  weights=[1 / len(train_dataset)] * len(train_dataset)).shuffle(BATCH_SIZE)
        self.write_tfrecords(oversampling_prefix, "train", resampled_ds)
        print("Total train instance after resampling:", self.num_counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--job-dir',
        default="./",
        help='Directory for staging and working files. '
             'This can be a Google Cloud Storage path.')
    parser.add_argument(
        '--job-type',
        required=True,
        type=str, choices=['discard', 'riichi', 'chi', 'pon', 'kan'],
        help='Type of pipeline needs to run')

    args = parser.parse_args()
    p = Pipeline(args.job_type)
    p.process(args.job_dir)