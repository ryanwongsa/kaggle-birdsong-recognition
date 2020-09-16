# Birdsong Recognition

## Instructions

### 0. (Optional) GCP VM Setup

```cmd
export IMAGE_FAMILY=pytorch-latest-gpu
export INSTANCE_NAME="pytorch-instance-2"
export ZONE="europe-west4-b"
export INSTANCE_TYPE="n1-standard-8"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=100GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible
```

```cmd
jupyter lab --ip=0.0.0.0
htop
```

### 1. Run on first install

Requires pytorch preinstalled. Skip installation of NVIDIA apex in the config if the gpu does not support mixed precision training.

```cmd
. startup.sh
```

### 2. Set Environment Variables

Both of these are optional:

```cmd
export NEPTUNE_API_TOKEN=<ACCESS_TOKEN>
export SLACK_URL=<SLACK_WEBHOOK_URL>
```

Neptune is not required, set `logger_name` in the config path to a different name and create an different logger

As Slack message notification will occur at the end of each epoch during training and validation with loss and metric information.

### 3. Training

```cmd
python train.py --config "config_params.local_parameters"
```

where `config` is the path to the `Parameter` class containing parameters for training. E.g. `config_params.local_parameters`
