# nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

# User-specific paths and settings
DATASET_PATH="<path/to/datasets>"  # e.g., "/home/datasets_root"
NAME="<experiment_name>"           # e.g., "box2video_experiment1"
OUT_DIR="<path/to/output>/${NAME}" # e.g., "/home/results/${NAME}"
PROJECT_NAME='<wandb_project_name>' # e.g., 'car_crash'
WANDB_ENTITY='<wandb_username>'    # Your Weights & Biases username
PRETRAINED_MODEL_PATH="<path/to/pretrained/model>" # e.g., "/path/to/pretrained/checkpoint"

# Create output directory
mkdir -p $OUT_DIR

# Save training script for reference
SCRIPT_PATH=$0
SAVE_SCRIPT_PATH="${OUT_DIR}/train_scripts.sh"
cp $SCRIPT_PATH $SAVE_SCRIPT_PATH
echo "Saved script to ${SAVE_SCRIPT_PATH}"

# Training command
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file config/a100l.yaml train_video_controlnet.py \
    --run_name $NAME \
    --data_root $DATASET_PATH \
    --project_name $PROJECT_NAME \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH \
    --output_dir $OUT_DIR \
    --variant fp16 \
    --dataset_name mmau \
    --train_batch_size 1 \
    --learning_rate 4e-5 \
    --checkpoints_total_limit 3 \
    --checkpointing_steps 300 \
    --checkpointing_time 10620 \
    --gradient_accumulation_steps 5 \
    --validation_steps 300 \
    --enable_gradient_checkpointing \
    --lr_scheduler constant \
    --report_to wandb \
    --seed 1234 \
    --mixed_precision fp16 \
    --clip_length 25 \
    --fps 6 \
    --min_guidance_scale 1.0 \
    --max_guidance_scale 3.0 \
    --noise_aug_strength 0.01 \
    --num_demo_samples 15 \
    --num_train_epochs 10 \
    --dataloader_num_workers 0 \
    --resume_from_checkpoint latest \
    --wandb_entity $WANDB_ENTITY \
    --train_H 320 \
    --train_W 512 \
    --use_action_conditioning \
    --contiguous_bbox_masking_prob 0.75 \
    --contiguous_bbox_masking_start_ratio 0.0 \
    --val_on_first_step
