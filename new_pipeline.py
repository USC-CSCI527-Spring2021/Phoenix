import argparse
import glob
import os
import random

import dill as pickle
import pandas as pd
import tensorflow as tf

from extract_features.FeatureGenerator import FeatureGenerator
from logs_parser import chi_pon_kan_model, discarded_model_dataset
from trainer.models import transform_discard_features

csv_path = glob.glob("./dataset/*.csv")


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
            num_train,
            num_val,
            input_feature_spec):
        self.num_train = num_train
        self.num_val = num_val
        self.input_feature_spec = input_feature_spec


class Pipeline():
    def __init__(self):
        self.counter = {
            "train": 0,
            "eval": 0
        }
        self.write_count = {
            "train": 0,
            "eval": 0
        }
        self.train = []
        self.eval = []
        self.log_count = 0
        self.params = {
            "discarded": {
                "table_bq_table": 'mahjong.discarded',
                "feature_spec": {
                    "features": tf.io.FixedLenFeature((16, 34, 1), tf.float32),
                    "labels": tf.io.FixedLenFeature((34,), tf.float32),
                },
                "process_fn": discarded_model_dataset.DiscardedFeatureExtractor(),
                "transform_fn": transform_discard_features,
            },
            "chi": {
                "table_bq_table": "mahjong.chi",
                "feature_spec": {
                    "features": tf.io.FixedLenFeature((63, 34, 1), tf.float32),
                    "labels": tf.io.FixedLenFeature((2,), tf.float32),
                },
                "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
                "transform_fn": FeatureGenerator().ChiFeatureGenerator,
            },
            "pon": {
                "table_bq_table": "mahjong.pon",
                "feature_spec": {
                    "features": tf.io.FixedLenFeature((63, 34, 1), tf.float32),
                    "labels": tf.io.FixedLenFeature((2,), tf.float32),
                },
                "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
                "transform_fn": FeatureGenerator().PonFeatureGenerator,
            },
            "kan": {
                "table_bq_table": "mahjong.kan",
                "feature_spec": {
                    "features": tf.io.FixedLenFeature((66, 34, 1), tf.float32),
                    "labels": tf.io.FixedLenFeature((2,), tf.float32),
                },
                "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
                "transform_fn": FeatureGenerator().KanFeatureGenerator,
            },
            "riichi": {
                "table_bq_table": "mahjong.riichi",
                "feature_spec": {
                    "features": tf.io.FixedLenFeature((62, 34, 1), tf.float32),
                    "labels": tf.io.FixedLenFeature((2,), tf.float32),
                },
                "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
                "transform_fn": FeatureGenerator().RiichiFeatureGenerator,
            }
        }

    def feature_writer(self, dataset_prefix, data):
        try:
            data_tuple = next(data)
            if data_tuple:
                if random.uniform(0, 100) < 100 - (0.8 * 10):
                    self.train.append(encoding(data_tuple).SerializeToString())
                    self.counter['train'] += 1
                else:
                    self.eval.append(encoding(data_tuple).SerializeToString())
                    self.counter['eval'] += 1

                if len(self.train) and self.counter['train'] % 40000 == 0:
                    writer = tf.data.experimental.TFRecordWriter(
                        "{}/train-dataset-{}".format(dataset_prefix, self.write_count["train"]))
                    writer.write(tf.data.Dataset.from_tensor_slices(self.train))
                    self.train = []
                    self.write_count['train'] += 1

                if len(self.eval) and self.counter['eval'] % 40000 == 0:
                    writer = tf.data.experimental.TFRecordWriter(
                        "{}/eval-dataset-{}".format(dataset_prefix, self.write_count["eval"]))
                    writer.write(tf.data.Dataset.from_tensor_slices(self.eval))
                    self.eval = []
                    self.write_count['eval'] += 1
        except StopIteration:
            pass

    def process(self, job_dir, job_type, csv_path):
        dataset_prefix = os.path.join(job_dir, 'processed_data', job_type)
        # delete existing processed data
        # if tf.io.gfile.exists(dataset_prefix):
        #     tf.io.gfile.rmtree(dataset_prefix)
        # tf.io.gfile.makedirs(dataset_prefix)

        for path in csv_path:
            df = pd.read_csv(path)
            logs_col = df["log_content"]
            for log_str in logs_col:
                self.log_count += 1
                extractors = chi_pon_kan_model.ChiPonKanFeatureExtractor().process(log_str)
                for data in extractors:
                    # self.feature_writer("riichi", fg.RiichiFeatureGenerator(data), self.riichi_features)
                    self.feature_writer(dataset_prefix, self.params[job_type]["transform_fn"](data))
                    # self.feature_writer("kan", fg.KanFeatureGenerator(data), self.kan_features)
                print("log #:", self.log_count)
            print(path, "Finished")
        with tf.io.gfile.GFile(os.path.join(dataset_prefix, job_type + "_meta"), 'wb') as f:
            pickle.dump(
                PreprocessData(self.counter['train'], self.counter['eval'], self.params[job_type]["feature_spec"]), f)
        print("# of Train:", self.counter['train'])
        print("# of Eval:", self.counter['eval'])


# with tf.io.gfile.GFile(os.path.join(dataset_prefix, "pon_meta"), 'wb') as f:
#     pickle.dump(
#         PreprocessData(3148529, 273726, params["pon"]["feature_spec"],
#                        train_dataset_dir + "*", eval_dataset_dir + "*"), f)
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
        type=str, choices=['discarded', 'riichi', 'chi', 'pon', 'kan'],
        help='Type of pipeline needs to run')
    p = Pipeline()
    args = parser.parse_args()

    p.process(args.job_dir, args.job_type, csv_path)
