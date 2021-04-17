
# Parse command line arguments
unset WORK_DIR
unset TYPE
MAX_DATA_FILES=5
PROJECT=$(gcloud config get-value project || echo $PROJECT)
REGION=us-central1
while [[ $# -gt 0 ]]; do
  case $1 in
    --training-type)
      TYPE=$2
      shift
      ;;
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

run gcloud ai-platform jobs submit training $TYPE_model_`date +"%Y%m%d_%H%M"` \
  --package-path trainer/ \
  --module-name trainer.task \
  --region $REGION \
  --python-version 3.7 \
  --runtime-version 2.4 \
  --job-dir $WORK_DIR \
  --config config.yaml \
  --stream-logs \
  -- \
  --model-type="discard" \
  --cloud-train=1

# discard model runner
  gcloud ai-platform jobs submit training discard_model_`date +"%Y%m%d_%H%M"` \
  --package-path trainer/ \
  --module-name trainer.task \
  --region us-central1 \
  --python-version 3.7 \
  --runtime-version 2.4 \
  --job-dir "gs://mahjong-dataset/" \
  --config trainer/config.yaml \
  --stream-logs \
  -- \
  --model-type="discard" \
  --cloud-train=1 \


# discard model tuner
  gcloud ai-platform jobs submit training discard_model_tuner`date +"%Y%m%d_%H%M"` \
  --package-path trainer/ \
  --module-name trainer.task \
  --region us-central1 \
  --python-version 3.7 \
  --runtime-version 2.4 \
  --job-dir "gs://mahjong-bucket/" \
  --config trainer/config.yaml \
  --stream-logs \
  -- \
  --model-type="discard" \
  --cloud-train=1 \
  --hypertune=1

# chi model runner
gcloud ai-platform jobs submit training chi_model_`date +"%Y%m%d_%H%M"` \
  --package-path trainer/ \
  --module-name trainer.task \
  --region us-central1 \
  --job-dir "gs://mahjong1/" \
  --config trainer/config.yaml \
  --stream-logs \
  -- \
  --model-type="chi" \
  --cloud-train=1

# riichi model runner
gcloud ai-platform jobs submit training riichi_model_`date +"%Y%m%d_%H%M"` \
  --package-path trainer/ \
  --module-name trainer.task \
  --region us-central1 \
  --job-dir "gs://mahjong-bucket/" \
  --config trainer/config.yaml \
  --stream-logs \
  -- \
  --model-type="riichi" \
  --cloud-train=1

# kan model runner
gcloud ai-platform jobs submit training kan_model_`date +"%Y%m%d_%H%M"` \
  --package-path trainer/ \
  --module-name trainer.task \
  --region us-central1 \
  --job-dir "gs://mahjong3/" \
  --config trainer/config.yaml \
  --stream-logs \
  -- \
  --model-type="kan" \
  --cloud-train=1

# pon model runner
gcloud ai-platform jobs submit training pon_model_`date +"%Y%m%d_%H%M"` \
  --package-path trainer/ \
  --module-name trainer.task \
  --region us-central1 \
  --job-dir "gs://mahjong3/" \
  --config trainer/config.yaml \
  --stream-logs \
  -- \
  --model-type="pon" \
  --cloud-train=1
# Runner for pipeline, replace the # to your information
# Local run only supply --job-type
  python pipeline.py --cloud=1 --job-dir=gs://mahjong-dataset \
  --job-type="riichi" --google-app-cred="#" --project="#" --region="#" --runner="DataflowRunner"

#local ai platform tester
  gcloud ai-platform local train \
  --distributed --worker-count 3 \
  --package-path trainer/  \
  --module-name trainer.task \
  -- \
  --model-type="discard" \
  --cloud-train=1