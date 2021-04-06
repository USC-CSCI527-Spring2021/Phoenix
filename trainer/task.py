import argparse
import datetime
import os
from time import sleep

import dill as pickle
import kerastuner as kt
import tensorflow as tf
from tensorflow import keras

from trainer.models import make_or_restore_model, scheduler, hypertune


def argument_parse():
    parser = argparse.ArgumentParser(description="Choose which model to train")
    parser.add_argument('--model-type', type=str, choices=['discarded', 'riichi', 'chi', 'pon', 'kan'])
    parser.add_argument('--cloud-train', type=int, default=0,
                        help='0 for local training, 1 for cloud training, default=20')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='number of times to go through the data, default=5000')
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
    parser.add_argument(
        '--hypertune',
        type=int,
        default=0,
        help='0 for no, 1 for hyper parameter tuning')
    parser.add_argument(
        '--class-weight',
        type=int,
        default=1,
        help='0 for oversampling, 1 for class-weight')
    args = parser.parse_args()
    return args


def _is_chief(cluster_resolver):
    task_type = cluster_resolver.task_type
    return task_type is None or task_type == 'chief'


def _get_temp_dir(model_path, cluster_resolver):
    worker_temp = f'worker{cluster_resolver.task_id}_temp'
    return os.path.join(model_path, worker_temp)


def save_model(model_path, model):
    # the following is need for TF 2.2. 2.3 onward, it can be accessed from strategy
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    is_chief = _is_chief(cluster_resolver)

    if not is_chief:
        model_path = _get_temp_dir(model_path, cluster_resolver)

    model.save(model_path)

    if is_chief:
        # wait for workers to delete; check every 100ms
        # if chief is finished, the training is done
        while tf.io.gfile.glob(os.path.join(model_path, "worker*")):
            sleep(0.1)

    if not is_chief:
        tf.io.gfile.rmtree(model_path)


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/jun/key.json"
    os.system("/sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | grep libcupti")

    args = argument_parse()
    from trainer.utils import CHECKPOINT_DIR, BATCH_SIZE, create_or_join

    BUFFER_SIZE = 10000
    if args.cloud_train:
        os.environ["TF_KERAS_RUNNING_REMOTELY"] = args.job_dir
        print("Start Training in Cloud")
        # tf_config = {
        #     "cluster": {
        #         "chief": ["10.128.0.1:3000"],
        #         "worker": ["10.128.0.2:3000",
        #                    "10.128.0.3:3000"],
        #     },
        #     "task": {"type": "chief", "index": 0}
        # }
        # os.environ["TF_CONFIG"] = json.dumps(tf_config)

        # communication_options = tf.distribute.experimental.CommunicationOptions(
        #     implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
        # strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
        # tf_config = json.dumps(os.environ["TF_CONFIG"])
        # num_workers = len(tf_config['cluster']['worker'])
        # print("Number of workers:", num_workers)
        strategy = tf.distribute.MirroredStrategy()
        # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        # BATCH_SIZE = BATCH_SIZE * num_workers

        BATCH_SIZE_PER_REPLICA = BATCH_SIZE
        BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
        # task_type, task_id = (strategy.cluster_resolver.task_type,
        #                       strategy.cluster_resolver.task_id)
        # write_model_path = write_filepath(os.path.join(create_or_join("models")), task_type, task_id)

    else:
        os.environ["TF_KERAS_RUNNING_REMOTELY"] = args.job_dir
        print("Start Training in Local")
        strategy = "local"

    checkpoint_path = create_or_join(os.path.join(CHECKPOINT_DIR, args.model_type))
    log_path = create_or_join("logs/" + args.model_type + timestamp)
    meta = tf.io.gfile.GFile(create_or_join('processed_data/{}/{}_meta'.format(args.model_type, args.model_type)), 'rb')
    preprocess_data = pickle.load(meta)

    class_weight = None
    num_classes = len(preprocess_data.classes_distribution)
    if args.class_weight:
        class_weight = {}
        for i in range(num_classes):
            class_weight[i] = preprocess_data.total / (num_classes * preprocess_data.classes_distribution[i])
        print('Weight for classes:', class_weight)
    else:
        # oversampling
        pass

    train_tfrecords = tf.io.gfile.glob(create_or_join("processed_data/{}/".format(args.model_type)) + "train-dataset*")
    val_tfrecords = tf.io.gfile.glob(create_or_join("processed_data/{}/".format(args.model_type)) + "eval-dataset*")
    train_dataset = tf.data.TFRecordDataset(train_tfrecords).map(
        lambda x: read_tfrecord(x, preprocess_data.input_feature_spec)) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = tf.data.TFRecordDataset(val_tfrecords).map(
        lambda x: read_tfrecord(x, preprocess_data.input_feature_spec)) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

    if args.hypertune:
        tuner = kt.Hyperband(
            hypermodel=hypertune,
            objective='val_categorical_accuracy',
            max_epochs=500,
            factor=2,
            hyperband_iterations=5,
            distribution_strategy=tf.distribute.MirroredStrategy(),
            directory=create_or_join("hyper_results_dir"),
            project_name='mahjong')
        print("hypertune: {}".format(args.model_type))
        tuner.search(train_dataset, epochs=args.num_epochs, validation_data=val_dataset,
                     validation_steps=1000,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy')])
    else:

        if args.model_type == 'discarded':
            input_shape = (16, 34, 1)
            model = make_or_restore_model(input_shape, args.model_type, strategy)
            callbacks = [
                keras.callbacks.TensorBoard(log_dir=log_path, update_freq='batch', histogram_freq=1),
                tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=create_or_join("model_backup")),
                keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy'),
                keras.callbacks.LearningRateScheduler(scheduler),
                keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                monitor='val_categorical_accuracy',
                                                save_freq=5000,
                                                )
            ]
        else:
            input_shape = (63, 34, 1)
            # steps_per_epoch = [20000, 5000, 5000, 5000]
            # validation_steps = [2500, 2500, 2500, 2500]
            # types = 0
            if args.model_type == 'chi':
                input_shape = (63, 34, 1)
            elif args.model_type == 'pon':
                input_shape = (63, 34, 1)
                # types = 1
                # generator = FG.PonFeatureGenerator()
            elif args.model_type == 'kan':
                input_shape = (66, 34, 1)
                # types = 2
            elif args.model_type == 'riichi':
                input_shape = (62, 34, 1)
                # types = 3
            model = make_or_restore_model(input_shape, args.model_type, strategy)
            callbacks = [
                keras.callbacks.TensorBoard(log_dir=log_path, update_freq='batch', histogram_freq=1),
                tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=create_or_join("model_backup")),
                keras.callbacks.EarlyStopping(monitor='accuracy'),
                keras.callbacks.LearningRateScheduler(scheduler),
                keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                monitor='val_accuracy',
                                                save_freq=1000,
                                                ),
            ]
        model.fit(train_dataset, epochs=args.num_epochs, validation_data=val_dataset,
                  # steps_per_epoch=preprocess_data.num_train // BATCH_SIZE,
                  class_weight=class_weight,
                  validation_steps=1000,
                  use_multiprocessing=True,
                  workers=-1,
                  callbacks=callbacks)
        model.save(os.path.join(create_or_join("models"), args.model_type))
        # save_model(os.path.join(create_or_join("models"), args.model_type), model)

    # model.save(write_model_path)
    # if args.cloud_train:
    #     # clean up for workers temp
    #     if not _is_chief(task_type, task_id):
    #         tf.io.gfile.rmtree(os.path.dirname(write_model_path))
    # sys.stdout.close()
