from __future__ import absolute_import

import argparse
# import base64
import csv
import random
from os import environ, path

import apache_beam as beam
import dill as pickle
import tensorflow as tf
import tensorflow_transform.beam as tft_beam
from apache_beam.options.pipeline_options import PipelineOptions
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata.schema_utils import schema_from_feature_spec

from extract_features.FeatureGenerator import FeatureGenerator
from logs_parser import discarded_model_dataset, chi_pon_kan_model
from trainer.config import TRAIN_SPLIT
from trainer.models import transform_discard_features


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


def data_pipeline(dataset_path, pipeline_opt, types, job_dir, project_id, **kwargs):
    """
    pipeline function
    :param dataset_path: the dataset path, e.g gs:/mahjong-dataset/dataset
    :param pipeline_opt: the pipline option function from apache beam
    :param types: what type of data is going to generate?
    :param job_dir: location for the job, e.g gs:/mahjong-dataset/
    :param project_id: project id of your account
    :param kwargs: additional args. check param dictionary in main function
    :return: no retrun
    """
    t = [str, dict, callable, callable]
    # check if correct kwargs pass in
    assert all([map(t[i], [v]) for i, (_, v) in enumerate(kwargs.items())]), "input type incorrect"
    # specify pipeline options
    p1 = beam.Pipeline(options=pipeline_opt)
    # required tmp location
    with tft_beam.impl.Context(temp_dir=path.join(job_dir, 'tmp')):
        # first step, read data line by line from the csv file
        # select 1 log and convert it to a list, [log_id,date,is_processed,was_error,log_content,log_hash]
        # extra [] is used to correctly parse the csv line
        # operation done in map, which is 1 on 1, 1 input -> 1 output
        dataset = (p1
                   | 'Read Data' >> beam.io.ReadFromText(dataset_path, skip_header_lines=True)
                   | 'Select one Log' >> beam.Map(
                    lambda x: [i for i in csv.reader([x], delimiter=',', doublequote=True)][0])
                   )

        if types == "discarded":
            # pass in the dataset variable to continue
            # go though process function, in this case,  discarded_model_dataset.DiscardedFeatureExtractor
            # this process function is a generator, i used the Pardo operation, a lower level operation compare to flatmap
            # output is 0 or many
            # next is to transform it to (x, 34, 1) shape, one hot for labels. in this case, transform_discard_features
            # 1 on 1 should be the case
            data = (
                    dataset
                    | "Process {} data".format(types) >> beam.ParDo(kwargs["process_fn"])
                    | "Transform {} data".format(types) >> beam.Map(kwargs["transform_fn"])
            )
        else:
            # similar to discarded, but different in transform_fn,
            # ChiFeatureGenerator, PonFeatureGenerator, KanFeatureGenerator
            # i faced some none data when above function is regular function, and generator function seem to solve it
            # I used flatmap because it's not 1 on 1, it might be 1 to 0
            data = (
                    dataset
                    | "Process {} data".format(types) >> beam.ParDo(kwargs["process_fn"])
                    | "Transform {} data".format(types) >> beam.FlatMap(kwargs["transform_fn"])
            )
        # meta data schema for tfrecord later
        input_meta = dataset_metadata.DatasetMetadata(schema_from_feature_spec(kwargs["feature_spec"]))
        # analyze the dataset for some stats
        # data, trans_func = (
        #         (data, input_meta)
        #         | 'Analyze data' >> tft_beam.AnalyzeAndTransformDataset(lambda x: x))
        # data, meta_data = data
        # split data to train split
        train_data, val_data = (
                data
                | "Split Data" >> beam.Partition(
            lambda x, _: int(random.uniform(0, 100) < 1 - TRAIN_SPLIT), 2)
        )

        # encode the numpy array to bytes and save to bigqueyr
        # write data to bigquery
        # _ = (
        #         (data)
        #         | "Encoding" >> beam.Map(lambda x: {"features": base64.b64encode(x["features"].tobytes()),
        #                                             "labels": base64.b64encode(x["labels"].tobytes())})
        #         | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(table=types,
        #                                                        # dataset in bigquery
        #                                                        dataset=kwargs["table_bq_table"].split(".")[0],
        #                                                        project=project_id,
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
        coder = example_proto_coder.ExampleProtoCoder(input_meta.schema)
        dataset_prefix = path.join(job_dir, 'processed_data/' + types)
        # delete existing processed data
        if tf.io.gfile.exists(dataset_prefix):
            tf.io.gfile.rmtree(dataset_prefix)
        tf.io.gfile.makedirs(dataset_prefix)

        train_dataset_dir = path.join(dataset_prefix, 'train-dataset')
        eval_dataset_dir = path.join(dataset_prefix, 'eval-dataset')
        (train_data
         | 'Write train dataset' >> beam.io.WriteToTFRecord(train_dataset_dir, coder))

        (val_data
         | 'Write val dataset' >> beam.io.WriteToTFRecord(eval_dataset_dir, coder))

        # Write the transform_fn
        # _ = (
        #     trans_func
        #     | 'Write transformFn' >> tft_beam.WriteTransformFn(path.join(dataset_prefix, '{}_transform'.format(types))))

        # write some information to later training
        with tf.io.gfile.GFile(path.join(dataset_prefix, types + "_meta"), 'wb') as f:
            pickle.dump(PreprocessData(kwargs["feature_spec"], train_dataset_dir + "*", eval_dataset_dir + "*"), f)
        # wait until finish
        p1.run().wait_until_finish()

def main(job_dir, job_type, project_id, region, google_app_cred, **kwargs):
    dataset_path = path.join(job_dir, "dataset/2021.csv")
    environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_app_cred
    # environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google.json"
    if kwargs['runner'] != 'DataflowRunner':
        pipeline_opt = PipelineOptions(direct_num_workers=0, direct_running_mode="multi_threading")
    else:
        pipeline_opt = PipelineOptions(runner=kwargs['runner'], project=project_id, region=region,
                                       setup_file="./setup.py",
                                       machine_type="n1-standard-8",
                                       # streaming=True,
                                       # max_num_workers=5,
                                       # experiments=['enable_stackdriver_agent_metrics'],
                                       # disk_size_gb=1000,
                                       # number_of_worker_harness_threads= 10,
                                       # experiments=['shuffle_mode=service', 'use_runner_v2'],
                                       # enable_streaming_engine=True,
                                       staging_location=path.join(job_dir, 'data-staging-tmp'),
                                       temp_location=path.join(job_dir, 'data-processing-tmp'),
                                       save_main_session=True)
    # pipeline_opt = PipelineOptions()
    # discarded_schema = "draw_tile:INTEGER,hands:STRING,discarded_tiles_pool:STRING," \
    #                    "four_players_open_hands:STRING,discarded_tile:INTEGER "
    params = {
        "discarded": {
            "table_bq_table": 'mahjong.discarded',
            "feature_spec": {
                "features": tf.io.FixedLenFeature((16, 34, 1), tf.int64),
                "labels": tf.io.FixedLenFeature((34,), tf.float32),
            },
            "process_fn": discarded_model_dataset.DiscardedFeatureExtractor(),
            "transform_fn": transform_discard_features,
        },
        "chi": {
            "table_bq_table": "mahjong.chi",
            "feature_spec": {
                "features": tf.io.FixedLenFeature((63, 34, 1), tf.int64),
                "labels": tf.io.FixedLenFeature((2,), tf.float32),
            },
            "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
            "transform_fn": FeatureGenerator().ChiFeatureGenerator,
        },
        "pon": {
            "table_bq_table": "mahjong.pon",
            "feature_spec": {
                "features": tf.io.FixedLenFeature((63, 34, 1), tf.int64),
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
                "features": tf.io.FixedLenFeature((62, 34, 1), tf.int64),
                "labels": tf.io.FixedLenFeature((2,), tf.float32),
            },
            "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
            "transform_fn": FeatureGenerator().RiichiFeatureGenerator,
        }
    }

    data_pipeline(dataset_path, pipeline_opt, job_type, job_dir, project_id, **params[job_type])


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
    parser.add_argument(
        '--project-id',
        default="",
        help='current project id'),
    parser.add_argument(
        '--region',
        default="",
        help='current region'),
    parser.add_argument(
        '--google-app-cred',
        default="",
        help='absolute path to your GOOGLE_APPLICATION_CREDENTIALS')
    args, pipeline_args = parser.parse_known_args()

    environ['TF_KERAS_RUNNING_REMOTELY'] = args.job_dir

    main(args.job_dir, args.job_type, project_id=args.project_id, region=args.region,
         google_app_cred=args.google_app_cred,
         runner="DataflowRunner" if args.cloud else "")
