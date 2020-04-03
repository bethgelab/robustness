#!/bin/bash

IMANGENETC_PATH="/ImageNet-C"
CHECKPOINT_PATH="/checkpoints"
cd /batchnorm || exit

export PYTHONPATH=/deps:/deps/robusta:$PYTHONPATH
# TODO for some reason this here does not work. Fix.
# pip install /deps || exit
python -c "print('Check import'); import robusta" || exit

process_split()
{
    local model=$1
    local checkpoint=$2
    local batchsize=$3
    local extra_args=$4
    local split=$5

    echo "Started for $model and split $split"
    echo "Checkpoint: $checkpoint"

    # It's ok if we use "--resume" but give an empty $checkpoint.
    # The adaptation module will ignore it.
    python src/evaluate.py \
                    --test-batch-size "$batchsize" \
                    --imagenet-path "$IMANGENETC_PATH/$split" \
                    --arch "$model" \
                    --pretrained \
                    --print-freq 1 \
                    --workers 15 \
                    --emission-path "$SCRATCH/run_experiment" \
                    --resume "$checkpoint" \
                    $extra_args
    
    echo "Successfully finished split $split"
}

iterate_through_imagenet_c()
{
    local model=$1
    local checkpoint=$2 
    local batchsize=$3
    local extra_args=$4

    # This array doesn't contain the hold-out corruptions:
    # gaussian_blur, saturate, spatter, speckle_noise
    local all_splits=("brightness"    "elastic_transform" "impulse_noise" 
                      "pixelate"      "snow"              "zoom_blur"
                      "contrast"      "fog"               "gaussian_noise" 
                      "jpeg_compression" "defocus_blur"   "frost" 
                      "glass_blur"    "motion_blur"       "shot_noise")

    # The dataset is split by corruption type into 15 splits.
    # These splits can be run in 2 ways:
    if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
        # 1. In a distributed environment (like Slurm), 
        # where each worker processes one split.
        touch "logs/$SLURM_ARRAY_TASK_ID.active"

        split_index=$SLURM_ARRAY_TASK_ID
        split=${all_splits[$split_index]}

        process_split "$model" "$checkpoint" "$batchsize" "$extra_args" "$split"

        rm "logs/$SLURM_ARRAY_TASK_ID.active"
    else
        # 2. Sequentially in a single process with a "for" loop;
        for split in "${all_splits[@]}" 
        do
            echo "Split $split started"  
            process_split "$model" "$checkpoint" "$batchsize" "$extra_args" "$split"
        done
    fi
}


# The row number is the same as the row number in Table 1 in the paper.
if [[ -n "$1" ]]; then
    row=$1
else
    echo 'You need to give a parameter for which row of Table 1 to compute'   
fi 


if [[ "$row" = "1" ]]; then
    model="resnet50"
    checkpoint="" # no checkpoint
    batchsize="512"
    iterate_through_imagenet_c $model "$checkpoint" $batchsize
fi

if [[ "$row" = "2" ]]; then
    model="resnet50"
    checkpoint="$CHECKPOINT_PATH/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar"
    batchsize="512"
    iterate_through_imagenet_c $model $checkpoint $batchsize
fi

if [[ "$row" = "3" ]]; then
    model="resnet50"
    checkpoint="$CHECKPOINT_PATH/ANT3x3Model.pth"
    batchsize="512"
    iterate_through_imagenet_c $model $checkpoint $batchsize
fi

if [[ "$row" = "4" ]]; then
    model="resnet50"
    checkpoint="$CHECKPOINT_PATH/ANT3x3_SIN_Model.pth"
    batchsize="512"
    iterate_through_imagenet_c $model $checkpoint $batchsize
fi

if [[ "$row" = "5" ]]; then
    model="resnet50"
    checkpoint="$CHECKPOINT_PATH/augmix_checkpoint.pth.tar"
    batchsize="512"
    iterate_through_imagenet_c $model $checkpoint $batchsize
fi

if [[ "$row" = "7" ]]; then
    model="resnet50"
    checkpoint="$CHECKPOINT_PATH/deepaugment.pth.tar"
    batchsize="512"
    iterate_through_imagenet_c $model $checkpoint $batchsize
fi

if [[ "$row" = "8" ]]; then
    model="resnet50"
    checkpoint="$CHECKPOINT_PATH/deepaugment_and_augmix.pth.tar"
    batchsize="512"
    iterate_through_imagenet_c $model $checkpoint $batchsize
fi

if [[ "$row" = "9" ]]; then
    model="resnext101_32x8d"
    checkpoint="$CHECKPOINT_PATH/resnext101_augmix_and_deepaugment.pth.tar"
    batchsize="128"
    iterate_through_imagenet_c $model $checkpoint $batchsize
fi

### Appendix: not part of the table anymore, but it is run in the same way

if [[ "$row" = "10" ]]; then
    model="efficientnet-b0"
    checkpoint="" # no checkpoint
    batchsize="128"
    iterate_through_imagenet_c $model "$checkpoint" $batchsize
fi

if [[ "$row" = "11" ]]; then
    model="resnet50"
    checkpoint="$CHECKPOINT_PATH/resnext101_augmix_and_deepaugment.pth.tar"
    batchsize="128"
    extra_args="--ema-batchnorm"
    iterate_through_imagenet_c $model $checkpoint $batchsize $extra_args
fi
