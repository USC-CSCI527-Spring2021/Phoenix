import os

GCP_BUCKET = 'mahjong-dataset'
BATCH_SIZE = 1
TRAIN_SPLIT = 0.8
checkpoint_dir = "checkpoints/"
BUCKET_PATH = "gs://" + GCP_BUCKET
JOB_DIR = os.environ.get("TF_KERAS_RUNNING_REMOTELY")


def get_root_path():
    if bool(JOB_DIR):
        return JOB_DIR
    else:
        return os.path.join(os.path.dirname(__file__), '..')


def create_or_join(dir_name):
    root = get_root_path()
    if bool(JOB_DIR):
        return os.path.join(JOB_DIR, dir_name)
    else:
        if not os.path.exists(os.path.join(root, dir_name)):
            os.makedirs(os.path.join(root, dir_name))
        return os.path.join(root, dir_name)
