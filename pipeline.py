from __future__ import absolute_import

import argparse
import csv
import random
from os import environ, path
from xml.etree import ElementTree as ET

import apache_beam as beam
import dill as pickle
import tensorflow as tf
# import tensorflow_transform.beam as tft_beam
from apache_beam.options.pipeline_options import PipelineOptions

# from tensorflow_transform.coders import example_proto_coder
# from tensorflow_transform.tf_metadata import dataset_metadata
# from tensorflow_transform.tf_metadata.schema_utils import schema_from_feature_spec
# import numpy as np
from extract_features.FeatureGenerator import FeatureGenerator
from logs_parser import discarded_model_dataset, chi_pon_kan_model
from trainer.models import transform_discard_features
from trainer.utils import TRAIN_SPLIT


class PreprocessData(object):
    """
    Serialized feature specs, output file path, so we can load it when training
    """

    def __init__(
            self,
            input_feature_spec,
            train_files_pattern,
            eval_files_pattern):
        self.input_feature_spec = input_feature_spec
        self.train_files_pattern = train_files_pattern
        self.eval_files_pattern = eval_files_pattern


class ValidateInputData(beam.DoFn):
    """This DoFn validates that every element matches the metadata given."""

    def __init__(self, feature_spec):
        self.feature_names = set(feature_spec.keys())

    def process(self, elem):
        if not isinstance(elem, dict):
            raise ValueError(
                'Element must be a dict(str, value). '
                'Given: {} {}'.format(elem, type(elem)))
        elem_features = set(elem.keys())
        if not self.feature_names.issubset(elem_features):
            raise ValueError(
                "Element features are missing from feature_spec keys. "
                'Given: {}; Features: {}'.format(
                    list(elem_features), list(self.feature_names)))
        yield elem


class DataToTfExampleDoFn(beam.DoFn):
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def process(self, feature_label_tuple):
        """
          Convert image to tf-example
        """
        feature, label = feature_label_tuple
        yield tf.train.Example(features=tf.train.Features(
            feature={'features': self._float_feature(feature.ravel().tolist()),
                     'labels': self._float_feature(label.ravel().tolist())
                     }))


def csv_parser(elem):
    log = [i for i in csv.reader([elem], delimiter=',', doublequote=True)][0][4]
    try:
        ET.fromstring(log)
        yield log
    except ET.ParseError:
        pass


def run(job_dir, job_type, beam_options):
    if beam_options and not isinstance(beam_options, PipelineOptions):
        raise ValueError(
            '`beam_options` must be {}. '
            'Given: {} {}'.format(PipelineOptions,
                                  beam_options, type(beam_options)))

    params = {
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

    t = [str, dict, callable, callable]
    # check if correct kwargs pass in
    assert all([map(t[i], [v]) for i, (_, v) in enumerate(params[job_type].items())]), "input type incorrect"
    dataset_path = path.join(job_dir, "dataset/*.csv")
    dataset_prefix = path.join(job_dir, 'processed_data/' + job_type)
    # delete existing processed data
    if tf.io.gfile.exists(dataset_prefix):
        tf.io.gfile.rmtree(dataset_prefix)
    tf.io.gfile.makedirs(dataset_prefix)

    train_dataset_dir = path.join(dataset_prefix, 'train-dataset')
    eval_dataset_dir = path.join(dataset_prefix, 'eval-dataset')
    # required tmp location
    with beam.Pipeline(options=beam_options) as p1:
        # with tft_beam.impl.Context(temp_dir=path.join(job_dir, 'tmp')):
        # first step, read data line by line from the csv file
        # select 1 log and convert it to a list, [log_id,date,is_processed,was_error,log_content,log_hash]
        # extra [] is used to correctly parse the csv line
        # operation done in map, which is 1 on 1, 1 input -> 1 output
        train_data, val_data = (p1
                                | 'Read Data' >> beam.io.ReadFromText(dataset_path, skip_header_lines=True)
                                | 'Select one Log' >> beam.FlatMap(csv_parser)
                                | "Process {} data".format(job_type) >> beam.ParDo(params[job_type]["process_fn"])
                                | "Transform {} data".format(job_type) >> beam.FlatMap(
                    (params[job_type]["transform_fn"]))
                                | "Encoding" >> beam.ParDo(DataToTfExampleDoFn())
                                | "SerialProtobuf" >> beam.Map(lambda x: x.SerializeToString())
                                | "Split Data" >> beam.Partition(
                    lambda x, _: int(random.uniform(0, 100) < 100 - (TRAIN_SPLIT * 10)), 2)
                                )

        # if job_type == "discarded":
        #     # pass in the dataset variable to continue
        #     # go though process function, in this case,  discarded_model_dataset.DiscardedFeatureExtractor
        #     # this process function is a generator, i used the Pardo operation, a lower level operation compare to flatmap
        #     # output is 0 or many
        #     # next is to transform it to (x, 34, 1) shape, one hot for labels. in this case, transform_discard_features
        #     # 1 on 1 should be the case
        #     data = (
        #             log
        #             | "Process {} data".format(job_type) >> beam.ParDo(params[job_type]["process_fn"])
        #             | "Transform {} data".format(job_type) >> beam.Map(params[job_type]["transform_fn"])
        #     )
        # else:
        #     # similar to discarded, but different in transform_fn,
        #     # ChiFeatureGenerator, PonFeatureGenerator, KanFeatureGenerator
        #     # i faced some none data when above function is regular function, and generator function seem to solve it
        #     # I used flatmap because it's not 1 on 1, it might be 1 to 0
        #     data = (
        #             log
        #             | "Process {} data".format(job_type) >> beam.ParDo(params[job_type]["process_fn"])
        #             | "Transform {} data".format(job_type) >> beam.ParDo(transmform_wrapper(params[job_type]["transform_fn"]))
        #             | "Validate Transform" >> beam.ParDo(ValidateInputData(params[job_type]["feature_spec"]))
        #     )
        # meta data schema for tfrecord later
        # input_meta = dataset_metadata.DatasetMetadata(schema_from_feature_spec(params[job_type]["feature_spec"]))
        # analyze the dataset for some stats
        # trans_func = (
        #         (data, input_meta)
        #         | 'Analyze data' >> tft_beam.AnalyzeDataset(lambda x: x)
        # )

        # data, meta_data = data

        # assert 0 < (TRAIN_SPLIT * 10) < 100, 'eval_percent must in the range (0-100)'
        # split data to train split
        # train_data, val_data = (
        #         data
        #         | "Split Data" >> beam.Partition(
        #     lambda x, _: int(random.uniform(0, 100) < 100 - (TRAIN_SPLIT * 10)), 2)
        # )
        # encode the numpy array to bytes and save to bigqueyr
        # write data to bigquery
        # _ = (
        #         (data)
        #         # | "Encoding" >> beam.Map(lambda x: {"features": base64.b64encode(x["features"].tobytes()),
        #         #                                     "labels": base64.b64encode(x["labels"].tobytes())})
        #         | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(table=job_type,
        #                                                        # dataset in bigquery
        #                                                        dataset=params[job_type]["table_bq_table"].split(".")[0],
        #                                                        project="project1-308005",
        #                                                        # schema of the bigquery table
        #                                                        schema="features:BYTES, labels:BYTES",
        #                                                        insert_retry_strategy='RETRY_ON_TRANSIENT_ERROR',
        #                                                        # streaming option
        #                                                        method="FILE_LOADS",
        #                                                        # temp file
        #                                                        custom_gcs_temp_location=job_dir + "/temps",
        #                                                        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        #                                                        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
        # )

        # example protobuf use to encode the data
        # coder = example_proto_coder.ExampleProtoCoder(input_meta.schema)

        (train_data
         | 'Write train dataset' >> beam.io.WriteToTFRecord(train_dataset_dir))

        (val_data
         | 'Write val dataset' >> beam.io.WriteToTFRecord(eval_dataset_dir))

        # # Write the transform_fn
        # _ = (
        #         trans_func
        #         | 'Write transformFn' >> tft_beam.WriteTransformFn(
        #     path.join(dataset_prefix, '{}_transform'.format(job_type))))
        with tf.io.gfile.GFile(path.join(dataset_prefix, job_type + "_meta"), 'wb') as f:
            pickle.dump(
                PreprocessData(params[job_type]["feature_spec"], train_dataset_dir + "*", eval_dataset_dir + "*"),
                f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--cloud',
        default=0,
        help='0 for local, 1 for cloud dataflow')
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
    # parser.add_argument(
    #     '--project-id',
    #     default="",
    #     help='current project id'),
    # parser.add_argument(
    #     '--region',
    #     default="",
    #     help='current region'),
    # parser.add_argument(
    #     '--runner',
    #     default="",
    #     help='current region'),
    parser.add_argument(
        '--google-app-cred',
        default="",
        help='absolute path to your GOOGLE_APPLICATION_CREDENTIALS')
    args, pipeline_args = parser.parse_known_args()
    environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_app_cred
    environ['TF_KERAS_RUNNING_REMOTELY'] = args.job_dir
    # pipeline_opt = PipelineOptions(direct_num_workers=0, direct_running_mode="multi_threading")
    pipeline_opt = PipelineOptions(pipeline_args,
                                   setup_file="./setup.py",
                                   # disk_size_gb=1000,
                                   # max_num_workers=2,
                                   #    use_public_ips=False,
                                   # machine_type="n1-highmem-96",
                                   # number_of_worker_harness_threads=30,
                                   experiments=["enable_execution_details_collection", "use_monitoring_state_manager"],
                                   # worker_harness_container_image="wate123/mahjong-dataflow:latest",
                                   staging_location=path.join(args.job_dir, 'data-staging-tmp'),
                                   temp_location=path.join(args.job_dir, 'data-processing-tmp'),
                                   save_main_session=True
                                   )
    # if pipeline_args.runner != 'DataflowRunner':

    # else:
    #     pipeline_opt = PipelineOptions(pipeline_args, save_main_session=True)
    # pipeline_opt = PipelineOptions(runner=kwargs['runner'], project=project_id, region=region,
    #                                setup_file="./setup.py",
    #                                # machine_type="n1-standard-8",
    #                                # streaming=True,
    #                                # max_num_workers=10,
    #                                experiments=['use_runner_v2', 'shuffle_mode=service'],
    #                                # disk_size_gb=200,
    #                                # number_of_worker_harness_threads= 10,
    #                                # experiments=['shuffle_mode=service', 'use_runner_v2'],
    #                                # enable_streaming_engine=True,
    #                                # worker_disk_type="compute.googleapis.com/projects/lithe-cursor-307422/zones/us-central1/diskTypes/pd-ssd",
    #                                # subnetwork="https://www.googleapis.com/compute/v1/projects/lithe-cursor-307422/regions/us-central1/subnetworks/mahjong",
    #                                # worker_harness_container_image= "wate123/mahjong-dataflow:latest",
    #                                staging_location=path.join(job_dir, 'data-staging-tmp'),
    #                                temp_location=path.join(job_dir, 'data-processing-tmp'),
    #                                save_main_session=True
    #                                )
    run(args.job_dir, args.job_type, pipeline_opt)
