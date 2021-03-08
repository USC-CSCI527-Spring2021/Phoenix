
# Parse command line arguments
unset WORK_DIR
MAX_DATA_FILES=5
PROJECT=$(gcloud config get-value project || echo $PROJECT)
REGION=us-central1
while [[ $# -gt 0 ]]; do
  case $1 in
    --work-dir)
      WORK_DIR=$2
      shift
      ;;
    --max-data-files)
      MAX_DATA_FILES=$2
      shift
      ;;
    --project)
      PROJECT=$2
      shift
      ;;
    --region)
      REGION=$2
      shift
      ;;
    *)
      echo "error: unrecognized argument $1"
      exit 1
      ;;
  esac
  shift
done

if [[ -z $WORK_DIR ]]; then
  echo "error: argument --work-dir is required"
  exit 1
fi

if [[ $WORK_DIR != gs://* ]]; then
  echo "error: --work-dir must be a Google Cloud Storage path"
  echo "       example: gs://mahjong-dataset"
  exit 1
fi

if [[ -z $PROJECT ]]; then
  echo 'error: --project is required to run in Google Cloud Platform.'
  exit 1
fi

# Wrapper function to print the command being run
function run {
  echo "$ $@"
  "$@"
}


export GOOGLE_APPLICATION_CREDENTIALS="/Users/junlin/key.json"

echo "Start Processing with Dataflow"
run python pipeline.py \
  --job_dir=$WORK_DIR \
  --cloud=1

echo "Submit Training Job to AI Platform"

run gcloud ai-platform jobs submit training discard_model_`date +"%Y%m%d_%H%M"` \
  --package-path trainer/ \
  --module-name trainer.task \
  --region $REGION \
  --python-version 3.7 \
  --runtime-version 2.4 \
  --job-dir $WORK_DIR \
  --stream-logs \
  -- \
  --model_type="discard" \
  --cloud_train=1