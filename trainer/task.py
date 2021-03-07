import argparse
import datetime
import os
from os import path

import dill as pickle
import tensorflow as tf
from tensorflow import keras

from extract_features.FeatureGenerator import FeatureGenerator
from trainer.models import make_or_restore_model, scheduler


def argument_parse():
    parser = argparse.ArgumentParser(description="Choose which model to train")
    parser.add_argument('--model_type', type=str, choices=['discard', 'riichi', 'chi', 'pon', 'kan'])
    parser.add_argument('--cloud_train', type=int, default=0,
                        help='0 for local training, 1 for cloud training, default=20')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
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


def read_tfrecord(serialized_example, types):
    with tf.io.gfile.GFile(create_or_join('processed_data/{}/{}_meta'.format(types, types), ), 'rb') as f:
        preprocess_data = pickle.load(f)
        example = tf.io.parse_example(serialized_example, preprocess_data.input_feature_spec)

        features = example['features']
        # labels = keras.utils.to_categorical(example['labels'], num_classes=34)
        labels = example['labels']
        return features, tf.cast(labels, dtype=tf.int64)


if __name__ == "__main__":
    # https://drive.google.com/uc\?id\=1iZpWSXRF9NlrLLwtxujLkAXk4k9KUgUN
    # f = h5py.File('logs_parser/discarded_model_dataset_sum_2021.hdf5', 'r')
    # states = f.get('hands')
    # labels = f.get('discarded_tile')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/junlin/key.json"

    args = argument_parse()
    from trainer.config import CHECKPOINT_DIR, BATCH_SIZE, create_or_join, get_root_path

    if args.cloud_train:
        os.environ["TF_KERAS_RUNNING_REMOTELY"] = args.job_dir
        print("Start Training in Cloud")
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

    # client = storage.Client()
    # bucket = client.get_bucket(GCP_BUCKET)

    if args.model_type == 'discard':
        # generator = DiscardFeatureGenerator(path.join(get_root_path(), "processed_data",
        #                                                     "discarded_model_summary.json"))
        train_tfrecords = tf.io.gfile.glob(create_or_join("processed_data/discarded/") + "train-dataset*")
        val_tfrecords = tf.io.gfile.glob(create_or_join("processed_data/discarded/") + "val-dataset*")
        train_dataset = tf.data.TFRecordDataset(train_tfrecords).map(lambda x: read_tfrecord(x, 'discarded')) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)
        val_dataset = tf.data.TFRecordDataset(val_tfrecords).map(lambda x: read_tfrecord(x, 'discarded'))

        input_shape = keras.Input((13, 34, 1))
        model = make_or_restore_model(input_shape, args.model_type)
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_path, update_freq='batch', histogram_freq=1),
            keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            save_weights_only=True,
                                            monitor='categorical_accuracy',
                                            save_freq="epoch",
                                            ),
            keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy'),
            keras.callbacks.LearningRateScheduler(scheduler)
        ]
    else:
        FG = FeatureGenerator(path.join(get_root_path(), "processed_data", "tiles_state_and_action_sum.json"))
        # if args.model_type == 'riichi':
        #     pass
        if args.model_type == 'chi':
            generator = FG.ChiFeatureGenerator()
        elif args.model_type == 'pon':
            generator = FG.PonFeatureGenerator()
        elif args.model_type == 'kan':
            generator = FG.KanFeatureGenerator()
        input_shape = keras.Input(((next(generator)[0].shape[0]), 34, 1))
        model = make_or_restore_model(input_shape, args.model_type)
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_path, update_freq='batch', histogram_freq=1),
            keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            save_weights_only=True,
                                            monitor='accuracy',
                                            save_freq="epoch",
                                            ),
            keras.callbacks.EarlyStopping(monitor='accuracy'),
            keras.callbacks.LearningRateScheduler(scheduler)
        ]
    # tf_dataset = tf.data.Dataset.from_generator(lambda: generator,
    #                                             (tf.int8, tf.int8)).shuffle(BATCH_SIZE, RANDOM_SEED, True)
    # train_dataset, val_dataset = split_dataset(tf_dataset)
    # sys.stderr = sys.stdout
    # sys.stdout = open('{}/{}.txt'.format('logs/stdout_logs', datetime.datetime.now().isoformat()), 'w')

    model.fit(train_dataset, epochs=args.num_epochs, validation_data=val_dataset, batch_size=BATCH_SIZE,
              use_multiprocessing=True,
              workers=-1,
              callbacks=callbacks)
    model.save(os.path.join(create_or_join("models"), args.model_type))
    # sys.stdout.close()
