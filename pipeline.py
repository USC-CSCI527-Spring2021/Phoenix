import argparse
import csv
import random
from os import environ, path

import apache_beam as beam
import dill as pickle
import tensorflow as tf
import tensorflow_transform.beam as tft_beam
from apache_beam.options.pipeline_options import PipelineOptions
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

from extract_features.FeatureGenerator import FeatureGenerator
from logs_parser import discarded_model_dataset, chi_pon_kan_model
from trainer.config import PROJECT_ID, REGION, TRAIN_SPLIT
from trainer.models import transform_discard_features


class PreprocessData(object):
    def __init__(
            self,
            input_feature_spec,
            train_files_pattern,
            eval_files_pattern):
        self.input_feature_spec = input_feature_spec
        self.train_files_pattern = train_files_pattern
        self.eval_files_pattern = eval_files_pattern


# class Csv_Parser(beam.DoFn):
#     def process(self, element, *args, **kwargs):
#         # data = [i for i in csv.reader([element], delimiter=',', doublequote=True)]
#         return element


def data_pipeline(dataset_path, pipeline_opt, types, job_dir, **kwargs):
    t = [dict, callable, callable]
    # check if correct type data pass in
    assert all([map(t[i], [v]) for i, (_, v) in enumerate(kwargs.items())]), "input type incorrect"
    with beam.Pipeline(options=pipeline_opt) as p1, tft_beam.impl.Context(temp_dir=path.join(job_dir, 'tmp')):

        # discarded_schema = "draw_tile:INTEGER,hands:STRING,discarded_tiles_pool:STRING," \
        #                    "four_players_open_hands:STRING,discarded_tile:INTEGER "

        dataset = (p1
                   | 'Read Data' >> beam.io.ReadFromText(dataset_path, skip_header_lines=True)
                   | 'Select one Log' >> beam.Map(
                    lambda x: [i for i in csv.reader([x], delimiter=',', doublequote=True)][0])
                   )
        if types == "discarded":
            data = (
                    dataset
                    | "Process {} data".format(types) >> beam.ParDo(kwargs["process_fn"])
                    | "Transform {} data".format(types) >> beam.Map(kwargs["transform_fn"])
            )
        elif types == "chi":
            data = (
                    dataset
                    | "Process {} data".format(types) >> beam.ParDo(kwargs["process_fn"])
                    | "Transform {} data".format(types) >> beam.ParDo(
                lambda x: kwargs["transform_fn"](x).ChiFeatureGenerator())
                    | "Filter None" >> beam.Filter(lambda x: x if x else False)
            )
        elif types == "pon":
            data = (
                    dataset
                    | "Process {} data".format(types) >> beam.ParDo(kwargs["process_fn"])
                    | "Transform {} data".format(types) >> beam.ParDo(
                lambda x: kwargs["transform_fn"](x).PonFeatureGenerator())
                    | "Filter None" >> beam.Filter(lambda x: x if x else False)
            )
        elif types == "kan":
            data = (
                    dataset
                    | "Process {} data".format(types) >> beam.ParDo(kwargs["process_fn"])
                    | "Transform {} data".format(types) >> beam.ParDo(
                lambda x: kwargs["transform_fn"](x).KanFeatureGenerator())
                    | "Filter None" >> beam.Filter(lambda x: x if x else False)
            )
        input_meta = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(kwargs["feature_spec"]))
        discard_data, trans_func = (
                (data, input_meta)
                | 'Analyze data' >> tft_beam.AnalyzeAndTransformDataset(lambda x: x))
        discard_data, meta_data = discard_data
        train_data, val_data = (
                discard_data
                | "Split Data" >> beam.Partition(
            lambda x, _: int(random.uniform(0, 100) < 1 - TRAIN_SPLIT), 2)
        )
        coder = example_proto_coder.ExampleProtoCoder(meta_data.schema)
        dataset_prefix = path.join(job_dir, 'processed_data/' + types)

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
        _ = (
                trans_func
                | 'Write transformFn' >> transform_fn_io.WriteTransformFn(
            path.join(dataset_prefix, '{}_transform'.format(types))))
        with tf.io.gfile.GFile(path.join(dataset_prefix, types + "_meta"), 'wb') as f:
            pickle.dump(PreprocessData(kwargs["feature_spec"], train_dataset_dir + "*", eval_dataset_dir + "*"), f,
                        protocol=1)


def main(job_dir, **kwargs):
    dataset_path = path.join(job_dir, "dataset/2022.csv")
    environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/junlin/key.json"
    if kwargs['runner'] != 'DataflowRunner':
        pipeline_opt = PipelineOptions(direct_num_workers=0, direct_running_mode="multi_threading")
    else:
        pipeline_opt = PipelineOptions(runner=kwargs['runner'], project=PROJECT_ID, region=REGION,
                                       setup_file="./setup.py",
                                       temp_location=path.join(job_dir, 'data-processing-tmp'),
                                       save_main_session=True)
    # pipeline_opt = PipelineOptions()
    # discarded_schema = "draw_tile:INTEGER,hands:STRING,discarded_tiles_pool:STRING," \
    #                    "four_players_open_hands:STRING,discarded_tile:INTEGER "
    params = {
        "discarded": {
            "feature_spec": {
                "features": tf.io.FixedLenFeature((13, 34, 1), tf.int64),
                "labels": tf.io.FixedLenFeature((34,), tf.float32),
            },
            "process_fn": discarded_model_dataset.DiscardedFeatureExtractor(),
            "transform_fn": transform_discard_features,
        },
        "chi": {
            "feature_spec": {
                "features": tf.io.FixedLenFeature((63, 34, 1), tf.int64),
                "labels": tf.io.FixedLenFeature((2,), tf.float32),
            },
            "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
            "transform_fn": FeatureGenerator,
        },
        "pon": {
            "feature_spec": {
                "features": tf.io.FixedLenFeature((63, 34, 1), tf.int64),
                "labels": tf.io.FixedLenFeature((2,), tf.float32),
            },
            "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
            "transform_fn": FeatureGenerator,
        },
        "kan": {
            "feature_spec": {
                "features": tf.io.FixedLenFeature((66, 34, 1), tf.int64),
                "labels": tf.io.FixedLenFeature((2,), tf.float32),
            },
            "process_fn": chi_pon_kan_model.ChiPonKanFeatureExtractor(),
            "transform_fn": FeatureGenerator,
        }
    }
    data_pipeline(dataset_path, pipeline_opt, "discarded", job_dir, **params["discarded"])
    data_pipeline(dataset_path, pipeline_opt, "pon", job_dir, **params["pon"])
    data_pipeline(dataset_path, pipeline_opt, "chi", job_dir, **params["chi"])
    data_pipeline(dataset_path, pipeline_opt, "kan", job_dir, **params["kan"])
    ###########################################
    # discarded pipeline
    # discard_feature_spec = {
    #     "features": tf.io.FixedLenFeature((13, 34, 1), tf.int64),
    #     "labels": tf.io.FixedLenFeature((34,), tf.float32),
    # }
    # data_pipeline(dataset, "discarded", discard_feature_spec,
    #               discarded_model_dataset.DiscardedFeatureExtractor(),
    #               transform_discard_features,
    #               job_dir
    #               )
    ###########################################
    # chi pipeline
    # chi_feature_spec = {
    #     "features": tf.io.FixedLenFeature((63, 34, 1), tf.int64),
    #     "labels": tf.io.FixedLenFeature((2,), tf.float32),
    # }
    # data_pipeline(dataset_path, pipeline_opt, "chi", chi_feature_spec,
    #               chi_pon_kan_model.ChiPonKanFeatureExtractor(),
    #               FeatureGenerator,
    #               job_dir
    #               )
    # ###########################################
    # # pon pipeline
    # pon_feature_spec = {
    #     "features": tf.io.FixedLenFeature((63, 34, 1), tf.int64),
    #     "labels": tf.io.FixedLenFeature((2,), tf.float32),
    # }
    # data_pipeline(dataset_path, pipeline_opt, "pon", pon_feature_spec,
    #               chi_pon_kan_model.ChiPonKanFeatureExtractor(),
    #               FeatureGenerator,
    #               job_dir
    #               )
    ###########################################
    # kan pipeline
    # kan_feature_spec = {
    #     "features": tf.io.FixedLenFeature((66, 34, 1), tf.int64),
    #     "labels": tf.io.FixedLenFeature((2,), tf.float32),
    # }
    # data_pipeline(dataset_path, pipeline_opt, "kan", kan_feature_spec,
    #               chi_pon_kan_model.ChiPonKanFeatureExtractor(),
    #               FeatureGenerator,
    #               job_dir
    #               )
    #
    # | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(DISCARD_TABLE_BQ, project=PROJECT_ID, schema=discarded_schema,
    #                                                method="FILE_LOADS",
    #                                                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
    #                                                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND))
    # | 'Write' >> beam.io.WriteToText('discard', '.txt'))
    #
    # with beam.Pipeline(options=pipeline_opt) as p2:
    #     feature_spec = {
    #         "features": tf.io.FixedLenFeature((63, 34, 1), tf.int64),
    #         "labels": tf.io.FixedLenFeature([], tf.int64),
    #     }
    #     dataset = (p1
    #                | 'Read Data' >> beam.io.ReadFromText(dataset_path, skip_header_lines=True)
    #                | 'Select one Log' >> beam.Map(
    #                 lambda x: [i for i in csv.reader([x], delimiter=',', doublequote=True)][0])
    #                | "Process Chi Pon Kan data" >> beam.ParDo(chi_pon_kan_model.DiscardedFeatureExtractor())
    #                | "Transform discard data" >> beam.Map(transform_discard_features))
    #     input_meta = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(feature_spec))
    #     dataset_and_metadata, transform_fn = (
    #             (dataset, input_meta)
    #             | 'Feature Scale' >> tft_beam.AnalyzeAndTransformDataset(
    #         lambda x: {"features": tft.scale_to_z_score(x['features'], tf.float32), "labels": x['labels']}))
    #     dataset, metadata = dataset_and_metadata
    #     train_data, val_data = (
    #             dataset
    #             | "Split Data" >> beam.Partition(
    #         lambda x, _: int(random.uniform(0, 100) < 1 - TRAIN_SPLIT), 2)
    #     )
    #     # coder = example_proto_coder.ExampleProtoCoder(metadata.schema)
    #     dataset_prefix = path.join(job_dir, 'processed_data')
    #
    #     (train_data
    #      | 'Write train dataset' >> beam.io.WriteToTFRecord(dataset_prefix))
    #
    #     (val_data
    #      | 'Write val dataset' >> beam.io.WriteToTFRecord(dataset_prefix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--cloud',
        default=0,
        help='0 for local, 1 for cloud dataflow')
    parser.add_argument(
        '--job-dir',
        # default="./",
        help='Directory for staging and working files. '
             'This can be a Google Cloud Storage path.')
    args, pipeline_args = parser.parse_known_args()

    environ['TF_KERAS_RUNNING_REMOTELY'] = args.job_dir
    # main(".", runner="DataflowRunner")
    main(args.job_dir, runner="DataflowRunner" if args.cloud else "")
