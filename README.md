# Cornell Birdcall Identification competition

1st Place solution to the Cornell Birdcall Identification competition.

[WIP: Detailed summary to be included]

## Instructions

### 0. (Optional) GCP VM Setup

```cmd
export IMAGE_FAMILY=pytorch-latest-gpu
export INSTANCE_NAME="pytorch-instance"
export ZONE="europe-west4-b"
export INSTANCE_TYPE="n1-standard-8"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=300GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible
```

### 1. Run on first install

Requires pytorch preinstalled. Skip installation of NVIDIA apex in the config if the gpu does not support mixed precision training.

```cmd
. startup.sh
```
NOTE: The sed training models don't support mixed precision training in the current state and will output NaN values. This can be fixed by adjusting the `amin` value but this is an untested change and may impact model performance.

### 2. Set Environment Variables

Both of these are optional:

```cmd
export NEPTUNE_API_TOKEN=<ACCESS_TOKEN>
export SLACK_URL=<SLACK_WEBHOOK_URL>
```

Neptune is not required, set `logger_name` in the config path to a different name and create an different logger.

As Slack message notification will occur at the end of each epoch during training and validation with loss.

### 3. Training

```cmd
python sed_train.py --config "config_params.example_config"
```
where `config` is the path to the `Parameter` class containing parameters for training. E.g. `config_params.example_config`.

Config files used for the file solution are available on request, as they currently contain personal neptune log details and directory structures.

### References

[1] [colorednoise](https://github.com/felixpatzelt/colorednoise)

[2] [audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)

[3] [imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler)

[4] [introduction-to-sound-event-detection](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection)

[5] [Pytorch-Audio-Emotion-Recognition](https://github.com/suicao/Pytorch-Audio-Emotion-Recognition)

[6] [argus-freesound](https://github.com/lRomul/argus-freesound)
