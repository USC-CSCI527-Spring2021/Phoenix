import os
import signal

import tensorflow as tf

# GCP_BUCKET = 'mahjong-dataset'
# GCP_BUCKET = 'mahjong-bucket'
GCP_BUCKET = 'mahjong1'
BATCH_SIZE = 64
TRAIN_SPLIT = 0.7
CHECKPOINT_DIR = "checkpoints"
RANDOM_SEED = 1
DISCARD_TABLE_BQ = "mahjong.discard"
CHI_PON_KAN_TABLE_BQ = "mahjong.chi_pon_kan"
# PROJECT_ID = "mahjong-305819"
# PROJECT_ID = "mahjong-307020"
PROJECT_ID = "lithe-cursor-307422"
REGION = "us-central1"


def get_root_path():
    JOB_DIR = os.environ.get("TF_KERAS_RUNNING_REMOTELY")
    if bool(JOB_DIR):
        return JOB_DIR
    else:
        l = os.path.dirname(__file__).split("/")
        l.pop()
        return "/".join(l)


def create_or_join(dir_name):
    JOB_DIR = os.environ.get("TF_KERAS_RUNNING_REMOTELY")
    root = get_root_path()
    if bool(JOB_DIR):
        dirs = os.path.join(JOB_DIR, dir_name)
        return dirs
    else:
        if not os.path.exists(os.path.join(root, dir_name)):
            os.makedirs(os.path.join(root, dir_name))
        return os.path.join(root, dir_name)


def read_tfrecord(serialized_example, input_spec):
    example = tf.io.parse_example(serialized_example, input_spec)
    features = example['features']
    labels = example['labels']
    return features, labels



class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True
