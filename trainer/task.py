import argparse
import datetime
import json
import os

import dill as pickle
import tensorflow as tf
from tensorflow import keras

from trainer.models import make_or_restore_model, scheduler


def argument_parse():
    parser = argparse.ArgumentParser(description="Choose which model to train")
    parser.add_argument('--model-type', type=str, choices=['discarded', 'riichi', 'chi', 'pon', 'kan'])
    parser.add_argument('--cloud-train', type=int, default=0,
                        help='0 for local training, 1 for cloud training, default=20')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='number of times to go through the data, default=10')
    # parser.add_argument(
    #     '--batch-size',
    #     default=128,
    #     type=int,
    #     help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models.',
        default=""
    )
    args = parser.parse_args()
    return args


def _is_chief(task_type, task_id):
    # Note: there are two possible `TF_CONFIG` configuration.
    #   1) In addition to `worker` tasks, a `chief` task type is use;
    #      in this case, this function should be modified to
    #      `return task_type == 'chief'`.
    #   2) Only `worker` task type is used; in this case, worker 0 is
    #      regarded as the chief. The implementation demonstrated here
    #      is for this case.
    # For the purpose of this colab section, we also add `task_type is None`
    # case because it is effectively run with only single worker.
    return (task_type == 'worker' and task_id == 0) or task_type is None


def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir


def write_filepath(filepath, task_type, task_id):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)


if __name__ == "__main__":
    # https://drive.google.com/uc\?id\=1iZpWSXRF9NlrLLwtxujLkAXk4k9KUgUN
    # f = h5py.File('logs_parser/discarded_model_dataset_sum_2021.hdf5', 'r')
    # states = f.get('hands')
    # labels = f.get('discarded_tile')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/jun/key.json"

    args = argument_parse()
    from trainer.config import CHECKPOINT_DIR, BATCH_SIZE, create_or_join

    if args.cloud_train:
        os.environ["TF_KERAS_RUNNING_REMOTELY"] = args.job_dir
        print("Start Training in Cloud")
        os.environ.pop('TF_CONFIG', None)
        tf_config = {
            "cluster": {
                "chief": ["{}-training-master:3000".format(args.model_type)],
                "worker": ["{}-training-worker1:3000".format(args.model_type),
                           "{}-training-worker2:3000".format(args.model_type)],
            },
            "task": {"type": "chief", "index": 0}
        }
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        communication_options = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
        strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
        num_workers = len(tf_config['cluster']['worker'])
        # strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        BUFFER_SIZE = 10000

        BATCH_SIZE_PER_REPLICA = 128
        BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_workers
        task_type, task_id = (strategy.cluster_resolver.task_type,
                              strategy.cluster_resolver.task_id)
        write_model_path = write_filepath(os.path.join(create_or_join("models")), task_type, task_id)

    else:
        os.environ["TF_KERAS_RUNNING_REMOTELY"] = args.job_dir
        print("Start Training in Local")

    # tfc.run(
    #     entry_point=None,
    #     requirements_txt="requirements.txt",
    #     distribution_strategy="auto",
    #     chief_config=tfc.MachineConfig(
    #         cpu_cores=8,
    #         memory=30,
    #         accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
    #         accelerator_count=2,
    #     ),
    #     docker_image_bucket_name=GCP_BUCKET,
    #     job_labels={"job": "discard_model"},
    #     stream_logs=True,
    # )

    checkpoint_path = create_or_join(os.path.join(CHECKPOINT_DIR, args.model_type)) + "/checkpoint"
    log_path = create_or_join("logs/" + timestamp)


    def read_tfrecord(serialized_example):
        with tf.io.gfile.GFile(create_or_join('processed_data/{}/{}_meta'.format(args.model_type, args.model_type), ),
                               'rb') as f:
            preprocess_data = pickle.load(f)
            example = tf.io.parse_example(serialized_example, preprocess_data.input_feature_spec)

            features = example['features']
            # labels = keras.utils.to_categorical(example['labels'], num_classes=34)
            labels = example['labels']
            return features, labels


    train_tfrecords = tf.io.gfile.glob(create_or_join("processed_data/{}/".format(args.model_type)) + "train-dataset*")
    val_tfrecords = tf.io.gfile.glob(create_or_join("processed_data/{}/".format(args.model_type)) + "eval-dataset*")
    train_dataset = tf.data.TFRecordDataset(train_tfrecords).map(read_tfrecord) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = tf.data.TFRecordDataset(val_tfrecords).map(read_tfrecord) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

    if args.model_type == 'discarded':
        # generator = DiscardFeatureGenerator(path.join(get_root_path(), "processed_data",
        #                                                     "discarded_model_summary.json"))
        input_shape = keras.Input((13, 34, 1))
        model = make_or_restore_model(input_shape, args.model_type, strategy)
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_path, update_freq='batch', histogram_freq=1),
            # keras.callbacks.ModelCheckpoint(checkpoint_path,
            #                                 save_weights_only=True,
            #                                 monitor='categorical_accuracy',
            #                                 save_freq="epoch",
            #                                 ),
            tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=create_or_join("/tmp/backup")),
            keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy'),
            keras.callbacks.LearningRateScheduler(scheduler)
        ]
    else:
        # FG = FeatureGenerator(path.join(get_root_path(), "processed_data", "tiles_state_and_action_sum.json"))
        # if args.model_type == 'riichi':
        #     pass
        input_shape = keras.Input((63, 34, 1))
        if args.model_type == 'chi':
            input_shape = keras.Input((63, 34, 1))
        elif args.model_type == 'pon':
            pass
            # generator = FG.PonFeatureGenerator()
        elif args.model_type == 'kan':
            input_shape = keras.Input((66, 34, 1))
            # generator = FG.KanFeatureGenerator()
        model = make_or_restore_model(input_shape, args.model_type, strategy)
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_path, update_freq='batch', histogram_freq=1),
            keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            save_weights_only=True,
                                            monitor='accuracy',
                                            save_freq='epoch',
                                            ),
            keras.callbacks.EarlyStopping(monitor='accuracy'),
            keras.callbacks.LearningRateScheduler(scheduler)
        ]

    # tf_dataset = tf.data.Dataset.from_generator(lambda: generator,
    #                                             (tf.int8, tf.int8)).shuffle(BATCH_SIZE, RANDOM_SEED, True)
    # train_dataset, val_dataset = split_dataset(tf_dataset)
    # sys.stderr = sys.stdout
    # sys.stdout = open('{}/{}.txt'.format('logs/stdout_logs', datetime.datetime.now().isoformat()), 'w')
    model.fit(train_dataset, epochs=args.num_epochs, validation_data=val_dataset, steps_per_epoch=100,
              use_multiprocessing=True,
              workers=-1,
              callbacks=callbacks)
    model.save(write_model_path)
    if args.cloud_train:
        # clean up for workers temp
        if not _is_chief(task_type, task_id):
            tf.io.gfile.rmtree(os.path.dirname(write_model_path))
    # sys.stdout.close()
