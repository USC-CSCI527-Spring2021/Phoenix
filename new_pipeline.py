import argparse
import glob
import os

import dill as pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from extract_features.FeatureGenerator import FeatureGenerator
from logs_parser import chi_pon_kan_model, discarded_model_dataset
from trainer.models import transform_discard_features
from trainer.utils import TRAIN_SPLIT, BATCH_SIZE


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


class Pipeline():
    def __init__(self, job_type):
        self.job_type = job_type
        self.counter = 0
        self.write_count = [0] * 34 if self.job_type == "discarded" else [0, 0]
        self.classes_distribution = [
                                        0] * 34 if self.job_type == "discarded" else [0, 0]
        # self.train = []
        # self.eval = []
        if self.job_type == "discarded":
            self.dataset = {i: [] for i in range(34)}
        else:
            self.dataset = {0: [], 1: []}
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

    def write_tfrecords(self, path_prefix, types=None, tfdata=None, train_test_val_total=None):
        # case 1: write multiple labels sepratetly
        if types is not None and not tfdata and not train_test_val_total:
            out_path = "{}/{}-dataset-{}".format(path_prefix, types, self.write_count[types])
            writer = tf.data.experimental.TFRecordWriter(out_path)
            writer.write(
                tf.data.Dataset.from_tensor_slices(self.dataset[types]))
        # case 2: write train, test or val dataset directly with sharding
        if tfdata and types and train_test_val_total:
            if train_test_val_total <= 40000:
                out_path = "{}/{}-dataset-1".format(path_prefix, types)
                writer = tf.data.experimental.TFRecordWriter(out_path)
                writer.write(tfdata)
            else:
                c = 0
                while train_test_val_total:
                    if train_test_val_total < 40000:
                        out_path = "{}/{}-dataset-{}".format(path_prefix, types, c)
                        writer = tf.data.experimental.TFRecordWriter(out_path)
                        writer.write(tfdata)
                    tmp = tfdata.take(40000)
                    out_path = "{}/{}-dataset-{}".format(path_prefix, types, c)
                    writer = tf.data.experimental.TFRecordWriter(out_path)
                    writer.write(tmp)
                    tfdata = tfdata.skip(40000)
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
        csv_path = glob.glob(os.path.join(job_dir, "dataset/*.csv"))
        # delete existing processed data
        if tf.io.gfile.exists(dataset_prefix):
            tf.io.gfile.rmtree(dataset_prefix)
        tf.io.gfile.makedirs(dataset_prefix)

        for path in csv_path:
            df = pd.read_csv(path)
            logs_col = df["log_content"]
            for log_str in logs_col:
                self.log_count += 1
                extractors = self.params[self.job_type]["process_fn"].process(
                    log_str)
                for data in extractors:
                    self.feature_writer(
                        dataset_prefix, self.params[self.job_type]["transform_fn"](data))
                print("log #:", self.log_count)
            print(path, "Finished")
        for cls in range(len(self.classes_distribution)):
            if not self.dataset[cls]:
                continue
            self.write_tfrecords(dataset_prefix, cls)
            # writer = tf.data.experimental.TFRecordWriter(
            #     "{}/{}-dataset-{}-f{}".format(dataset_prefix, cls, self.classes_distribution[cls],
            #                                   self.write_count[cls]))
            # writer.write(
            #     tf.data.Dataset.from_tensor_slices(self.dataset[cls]))
        meta_data = PreprocessData(self.counter, self.classes_distribution,
                                   self.params[self.job_type]["feature_spec"])
        with tf.io.gfile.GFile(os.path.join(dataset_prefix, self.job_type + "_meta"), 'wb') as f:
            pickle.dump(meta_data, f)

        print("Classes distribution:", self.classes_distribution)

        meta_data = pickle.load(open(os.path.join(dataset_prefix, "{}_meta".format(self.job_type)), 'rb'))
        train_size = int(TRAIN_SPLIT * meta_data.total)
        val_size = int((1 - TRAIN_SPLIT) / 2 * meta_data.total)
        test_size = int((1 - TRAIN_SPLIT) / 2 * meta_data.total)

        oversampling_prefix = os.path.join(job_dir, 'with_oversampling_data', self.job_type)
        if tf.io.gfile.exists(oversampling_prefix):
            tf.io.gfile.rmtree(oversampling_prefix)
        tf.io.gfile.makedirs(oversampling_prefix)

        tfrecords = glob.glob(os.path.join("./processed_data", self.job_type, "*-*"))
        full_dataset = tf.data.TFRecordDataset(tfrecords).shuffle(BATCH_SIZE)

        train_dataset = full_dataset.take(train_size)
        test_dataset = full_dataset.skip(train_size)
        val_dataset = test_dataset.skip(val_size)
        test_dataset = test_dataset.take(test_size)

        self.write_tfrecords(oversampling_prefix, "val", val_dataset, val_size)
        self.write_tfrecords(oversampling_prefix, "test", test_dataset, test_size)

        final_dataset = [[] for _ in range(len(self.classes_distribution))]
        tot = 0
        for raw in train_dataset:
            raw_dict = tf.io.parse_example(raw, meta_data.input_feature_spec)
            label = raw_dict['labels']
            l = int(np.argmax(label))
            tot += 1
            final_dataset[l].append(raw)
        final_dataset = [tf.data.Dataset.from_tensor_slices(data) for data in final_dataset]
        resampled_ds = tf.data.experimental.sample_from_datasets(final_dataset,
                                                                 weights=[1 / len(final_dataset)] * len(final_dataset))
        self.write_tfrecords(oversampling_prefix, "train", resampled_ds, tot)


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

    args = parser.parse_args()
    p = Pipeline(args.job_type)
    p.process(args.job_dir)

    # oversampling using ADASYN
    # preprocess_data=pickle.load(open(os.path.join(
    #     "./processed_data", args.job_type, args.job_type + "_meta"), "rb"))
    # train_tfrecords=glob.glob(os.path.join(
    #     "./processed_data", args.job_type, "train-dataset*"))
    # val_tfrecords=glob.glob(os.path.join(
    #     "./processed_data", args.job_type, "eval-dataset*"))
    #
    # train_dataset=tf.data.TFRecordDataset(train_tfrecords).map(
    #     lambda x: read_tfrecord(x, preprocess_data.input_feature_spec))
    # val_dataset=tf.data.TFRecordDataset(val_tfrecords).map(
    #     lambda x: read_tfrecord(x, preprocess_data.input_feature_spec))
    # whole_dataset=np.vstack(
    #     (list(train_dataset.take(10)), list(val_dataset.take(10))))
    # x, y=np.array(list(map(lambda i: i.numpy(), whole_dataset[:, 0]))), np.array(
    #     list(map(lambda i: i.numpy(), whole_dataset[:, 1])))
    # ADASYN().fit_resample(x, y)
