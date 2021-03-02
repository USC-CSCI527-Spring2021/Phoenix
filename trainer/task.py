import argparse
import datetime
import os
from os import path

import tensorflow as tf
from tensorflow import keras

from extract_features.FeatureGenerator import FeatureGenerator
from logs_parser import discarded_model_dataset, chi_pon_kan_model
from trainer.config import checkpoint_dir, BATCH_SIZE, create_or_join
from trainer.models import DiscardFeatureGenerator, make_or_restore_model, scheduler


def argument_parse():
    parser = argparse.ArgumentParser(description="Choose which model to train")
    parser.add_argument('--model_type', type=str, choices=['discard', 'riichi', 'chi', 'pon', 'kan'])
    parser.add_argument('--cloud_train', type=int, default=0,
                        help='0 for local training, 1 for cloud training, default=20')
    # parser.add_argument(
    #     '--num-epochs',
    #     type=int,
    #     default=20,
    #     help='number of times to go through the data, default=20')
    # parser.add_argument(
    #     '--batch-size',
    #     default=128,
    #     type=int,
    #     help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models.',
        required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # https://drive.google.com/uc\?id\=1iZpWSXRF9NlrLLwtxujLkAXk4k9KUgUN
    # f = h5py.File('logs_parser/discarded_model_dataset_sum_2021.hdf5', 'r')
    # states = f.get('hands')
    # labels = f.get('discarded_tile')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/jun/key.json"

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

    args = argument_parse()
    if args.cloud_train:
        os.environ["TF_KERAS_RUNNING_REMOTELY"] = args.job_dir
        print("Start Training in Cloud")
    else:
        os.environ["TF_KERAS_RUNNING_REMOTELY"] = args.job_dir
        print("Start Training in Local")

    if not os.listdir(create_or_join("processed_data")):
        discarded_model_dataset.main()
        chi_pon_kan_model.main()

    checkpoint_path = create_or_join(checkpoint_dir + "/{}".format(args.model_type)) + "/checkpoint"
    log_path = create_or_join("logs/" + timestamp)

    # client = storage.Client()
    # bucket = client.get_bucket(GCP_BUCKET)

    if args.model_type == 'discard':
        train_generator = DiscardFeatureGenerator(path.join(create_or_join("processed_data"),
                                                            "discarded_model_summary.json"))
        val_generator = DiscardFeatureGenerator(path.join(create_or_join("processed_data"),
                                                          "discarded_model_summary.json"), False)
        input_shape = keras.Input((next(train_generator)[0].shape[0], 34, 1))
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
        FG = FeatureGenerator(path.join(create_or_join("processed_data"), "tiles_state_and_action_sum.json"))
        # if args.model_type == 'riichi':
        #     pass
        if args.model_type == 'chi':
            train_generator = FG.ChiFeatureGenerator()
        elif args.model_type == 'pon':
            train_generator = FG.PonFeatureGenerator()
        elif args.model_type == 'kan':
            train_generator = FG.KanFeatureGenerator()
        input_shape = keras.Input(((next(train_generator)[0].shape[0]), 34, 1))
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

    # states = np.array([s[:1000] for s in states])

    # labels = labels[:1000]
    # print(states.shape)
    train_dataset = tf.data.Dataset.from_generator(lambda: train_generator,
                                                   (tf.int8, tf.int8)).shuffle(
        BATCH_SIZE, 1, True)
    val_dataset = tf.data.Dataset.from_generator(lambda: val_generator,
                                                 (tf.int8, tf.int8)).shuffle(BATCH_SIZE, 1, True)

    # sys.stderr = sys.stdout
    # sys.stdout = open('{}/{}.txt'.format('logs/stdout_logs', datetime.datetime.now().isoformat()), 'w')

    model.fit(train_dataset, epochs=50, batch_size=BATCH_SIZE, use_multiprocessing=True,
              workers=-1,
              callbacks=callbacks)
    model.save(create_or_join("models") + "/" + args.model_type)
    # sys.stdout.close()
