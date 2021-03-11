import argparse
import base64
import csv
from os import environ, path

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform.beam as tft_beam
from apache_beam.options.pipeline_options import PipelineOptions

from extract_features.FeatureGenerator import FeatureGenerator
from logs_parser import discarded_model_dataset, chi_pon_kan_model
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


def data_pipeline(dataset_path, pipeline_opt, types, job_dir, project_id, **kwargs):
    t = [str, dict, callable, callable]
    # check if correct type data pass in
    assert all([map(t[i], [v]) for i, (_, v) in enumerate(kwargs.items())]), "input type incorrect"
    with beam.Pipeline(options=pipeline_opt) as p1, tft_beam.impl.Context(temp_dir=path.join(job_dir, 'tmp')):
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
        else:
            data = (
                    dataset
                    | "Process {} data".format(types) >> beam.ParDo(kwargs["process_fn"])
                    | "Transform {} data".format(types) >> beam.FlatMap(kwargs["transform_fn"])
            )
        # input_meta = dataset_metadata.DatasetMetadata(schema_from_feature_spec(kwargs["feature_spec"]))
        #
        # data, trans_func = (
        #         (data, input_meta)
        #         | 'Analyze data' >> tft_beam.AnalyzeAndTransformDataset(lambda x: x))
        # data, meta_data = data
        # train_data, val_data = (
        #         data
        #         | "Split Data" >> beam.Partition(
        #     lambda x, _: int(random.uniform(0, 100) < 1 - TRAIN_SPLIT), 2)
        # )

        _ = (
                (data)
                | "Encoding" >> beam.Map(lambda x: {"features": base64.b64encode(x["features"].tobytes()),
                                                    "labels": base64.b64encode(x["labels"].tobytes())})
                | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(table=types,
                                                               dataset=kwargs["table_bq_table"].split(".")[0],
                                                               project=project_id,
                                                               schema="features:BYTES, labels:BYTES",
                                                               insert_retry_strategy='RETRY_ON_TRANSIENT_ERROR',
                                                               method="FILE_LOADS",
                                                               custom_gcs_temp_location=job_dir + "/temps",
                                                               create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                                                               write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
        )

        # coder = example_proto_coder.ExampleProtoCoder(input_meta.schema)
        # dataset_prefix = path.join(job_dir, 'processed_data/' + types)
        #
        # if tf.io.gfile.exists(dataset_prefix):
        #     tf.io.gfile.rmtree(dataset_prefix)
        # tf.io.gfile.makedirs(dataset_prefix)
        # train_dataset_dir = path.join(dataset_prefix, 'train-dataset')
        # eval_dataset_dir = path.join(dataset_prefix, 'eval-dataset')
        # (train_data
        #  | 'Write train dataset' >> beam.io.WriteToTFRecord(train_dataset_dir, coder))
        #
        # (val_data
        #  | 'Write val dataset' >> beam.io.WriteToTFRecord(eval_dataset_dir, coder))
        #
        # # Write the transform_fn
        # _ = (
        #     trans_func
        #     | 'Write transformFn' >> tft_beam.WriteTransformFn(path.join(dataset_prefix, '{}_transform'.format(types))))
        # with tf.io.gfile.GFile(path.join(dataset_prefix, types + "_meta"), 'wb') as f:
        #     pickle.dump(PreprocessData(kwargs["feature_spec"], train_dataset_dir + "*", eval_dataset_dir + "*"), f)


def main(job_dir, job_type, project_id, region, google_app_cred, **kwargs):
    dataset_path = path.join(job_dir, "dataset/*.csv")
    environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_app_cred
    # environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google.json"
    if kwargs['runner'] != 'DataflowRunner':
        pipeline_opt = PipelineOptions(direct_num_workers=0, direct_running_mode="multi_threading")
    else:
        pipeline_opt = PipelineOptions(runner=kwargs['runner'], project=project_id, region=region,
                                       setup_file="./setup.py",
                                       # machine_type="n1-highmem-8",
                                       # streaming=True,
                                       experiments=['use_runner_v2', 'shuffle_mode=service'],
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
            "process_fn": discarded_model_dataset.DiscardedFeatureExtractor,
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
    # data_pipeline(dataset_path, pipeline_opt, "discarded", job_dir, **params["discarded"])
    # data_pipeline(dataset_path, pipeline_opt, "pon", job_dir, **params["pon"])
    # data_pipeline(dataset_path, pipeline_opt, "chi", job_dir, **params["chi"])
    # data_pipeline(dataset_path, pipeline_opt, "kan", job_dir, **params["kan"])
    # data_pipeline(dataset_path, pipeline_opt, "riichi", job_dir, **params["riichi"])
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
    # main(".", runner="DataflowRunner")
    main(args.job_dir, args.job_type, project_id=args.project_id, region=args.region,
         google_app_cred=args.google_app_cred,
         runner="DataflowRunner" if args.cloud else "")
