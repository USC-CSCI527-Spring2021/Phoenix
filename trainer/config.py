import os

GCP_BUCKET = 'mahjong-dataset'
BATCH_SIZE = 1
TRAIN_SPLIT = 0.8
CHECKPOINT_DIR = "checkpoints"
RANDOM_SEED = 1
DISCARD_TABLE_BQ = "mahjong.discarded"
CHI_PON_KAN_TABLE_BQ = "mahjong.chi_pon_kan"
PROJECT_ID = "mahjong-305819"
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
