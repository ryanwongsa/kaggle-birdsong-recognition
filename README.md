# Cornell Birdcall Identification Competition

1st Place solution to the [Cornell Birdcall Identification competition](https://www.kaggle.com/c/birdsong-recognition) hosted on Kaggle.

## Context

> In this competition, you will identify a wide variety of bird vocalizations in soundscape recordings. Due to the complexity of the recordings, they contain weak labels. There might be anthropogenic sounds (e.g., airplane overflights) or other bird and non-bird (e.g., chipmunk) calls in the background, with a particular labeled bird species in the foreground. Bring your new ideas to build effective detectors and classifiers for analyzing complex soundscape recordings!

## Evaluation

> The hidden test_audio directory contains approximately 150 recordings in mp3 format, each roughly 10 minutes long. They will not all fit in a notebook's memory at the same time. The recordings were taken at three separate remote locations in North America. Sites 1 and 2 were labeled in 5 second increments and need matching predictions, but due to the time consuming nature of the labeling process the site 3 files are only labeled at the file level. Accordingly, site 3 has relatively few rows in the test set and needs lower time resolution predictions.

Scores were evaluated based on their row-wise micro averaged F1 score.

## Solution

My approach used a Sound Event Detection approach described in [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211). Kaggle user [hidehisaarai1213](hidehisaarai1213) provides a good explaination of how it works in the Kaggle kernel [introduction to sound event detection](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection).

### Main Model Differences

- Switched the CNN Feature extractor with a pretrained Densenet121 model
- Replaced the `torch.clamp` method with `torch.tanh` in the attention layer.
- Reduced the AttBlock size to 1024.

The reasons for making the changes was to avoid overfitting because of the lack of data with less than or equal to 100 labelled samples per bird class.

### Data Augmentation

[Audiomentation library](https://github.com/iver56/audiomentations) provided an easy way to add data augmentation to the audio samples. The following augmentation was applied during training:

- AddGaussianNoise
- AddGaussianSNR
- Gain
- AddBackgroundNoise (based on pre-generated 1 minute samples of pink noise)
- AddShortNoises (based on pre-generated 1 minute samples of pink noise)

## Inference

Model ensembling by voting and thresholds on both `clipwise_output` and `framewise_output` was key to reducing the number of false positives and maximising the f1-score.

- 4 fold models (without mixup)
- 5 fold models (without mixup)
- 4 fold models (with mixup)

2 submissions were allowed to be selected before the Private Leaderboard was revealed. My top ensemble was if 4 out of the 13 models predicted a bird with a threshold of `0.3` for both `clipwise_output` and `framewise_output` was within the audio snippet then it would be accepted as a valid prediction. This ensemble was not a the highest model on the public leaderboard but I selected it based on what I felt would be the most confident in as 3 votes felt too low for 13 models (highest on Public Leaderboard). With this ensemble I was able to jump from 7th on the Public Leaderboard to 1st on the Private Leaderboard.

My second selection was based on 9 models (the models without mixup). My top scoring ensemble with 9 models was with 3 votes and a threshold of `0.5` for the frame threshold and `0.3` for the clip threshold. I didn't try a threshold of `0.3` for the frame threshold since I ran out of submissions (max 2 submissions per day). The selected 9 model ensemble would have put me at 3rd position on the Private Leaderboard.

| **13 Models**        | Clip Threshold | Frame Threshold | Public Leaderboard | Private Leaderboard | Selected |
|---------|----------------|-----------------|--------------------|---------------------|----------|
| 4 votes | 0.3            | 0.3             | 0.616              | 0.681               | x        |
| 4 votes | 0.3            | 0.5             | 0.615              | 0.679               |         |
| 5 votes | 0.3            | 0.3             | 0.609              | 0.679               |         |
| 5 votes | 0.3            | 0.5             | 0.606              | 0.676               |         |
| 3 votes | 0.3            | 0.3             | 0.617              | 0.679               |         |
| 3 votes | 0.3            | 0.5             | 0.614              | 0.679               |         |

| **9 Models**        | Clip Threshold | Frame Threshold | Public Leaderboard | Private Leaderboard | Selected |
|---------|----------------|-----------------|--------------------|---------------------|----------|
| 3 votes | 0.3            | 0.5             | 0.613              | 0.676               | x        |
| 2 votes | 0.3            | 0.5             | 0.614              | 0.669               |         |
| 4 votes | 0.3            | 0.5             | 0.610              | 0.675               |         |

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

NOTE: The sed training models don't support mixed precision training in the current state and will output NaN values. This can be fixed by adjusting the `amin` value but this is an untested change and may impact model performance. `USE_AMP` in the `base_engine.py` will also need to be set to `True` (currently disabled even if amp is installed).

### 2. Set Environment Variables

Both of these are optional:

```cmd
export NEPTUNE_API_TOKEN=<ACCESS_TOKEN>
export SLACK_URL=<SLACK_WEBHOOK_URL>
```

Neptune is not required, set `logger_name` in the config path to a different name and create an different logger.

A Slack message notification will occur at the end of each epoch during training and validation with loss.

### 3. Training


```cmd
python sed_train.py --config "config_params.example_config"
```
where `config` is the path to the `Parameter` class containing parameters for training. E.g. `config_params.example_config`.

Config files used for the final solution are available in the `src/config_params/final_sed` and `src/config_params/final_sed_5_fold` folder.

Note: training was completed on the resampled 32kHz wav equivalent of the training data provided on Kaggle, with the same folder structure. In order for the given configs to work the following instructions need to be followed
- store the training data in `/data`
- store the generated pink noise in `/pinknoise`
- `/background/data_ssw` in the validation dataloader is data provided in the discuss [here](https://www.kaggle.com/c/birdsong-recognition/discussion/158877#911336), with 5-30 second clips extracted where no bird call was found during the extracted time. Adding this is not required.
- change the `project_name` in the config files to your personal neptune project name if using a neptune logger.

## Kaggle Solution

- Kaggle Kernel used to generate pink noise: https://www.kaggle.com/taggatle/noise-generator
- Example Training on Kaggle Kernel: https://www.kaggle.com/taggatle/example-training-notebook
- Final Trained Models: https://www.kaggle.com/taggatle/birdsongdetectionfinalsubmission1
- Kaggle Kernel which achieves 1st Place: https://www.kaggle.com/taggatle/cornell-birdcall-identification-1st-place-solution

## References

[1] [colorednoise](https://github.com/felixpatzelt/colorednoise)

[2] [audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)

[3] [imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler)

[4] [introduction-to-sound-event-detection](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection)

[5] [Pytorch-Audio-Emotion-Recognition](https://github.com/suicao/Pytorch-Audio-Emotion-Recognition)

[6] [argus-freesound](https://github.com/lRomul/argus-freesound)
