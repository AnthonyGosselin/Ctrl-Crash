# nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

# User-specific paths and settings
DATASET_PATH="<path/to/datasets>"  # e.g., "/home/datasets_root"
NAME="<experiment_name>"           # e.g., "box2video_experiment1"
OUT_DIR="<path/to/output>/${NAME}" # e.g., "/home/results/${NAME}"
PROJECT_NAME='<wandb_project_name>' # e.g., 'car_crash'
WANDB_ENTITY='<wandb_username>'    # Your Weights & Biases usernames
PRETRAINED_MODEL_PATH="stabilityai/stable-video-diffusion-img2vid-xt" # HuggingFace model ID

# export HF_HOME=/path/to/root # Where the SVD pretrained models are/will be downloaded

# Create output directory
mkdir -p $OUT_DIR

# Save training script for reference
SCRIPT_PATH=$0
SAVE_SCRIPT_PATH="${OUT_DIR}/train_scripts.sh"
cp $SCRIPT_PATH $SAVE_SCRIPT_PATH
echo "Saved script to ${SAVE_SCRIPT_PATH}"

# Training command
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file config/a100l.yaml train_video_diffusion.py \
    --run_name $NAME \
    --data_root $DATASET_PATH \
    --project_name $PROJECT_NAME \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH \
    --output_dir $OUT_DIR \
    --variant fp16 \
    --dataset_name mmau \
    --train_batch_size 1 \
    --learning_rate 1e-5 \
    --checkpoints_total_limit 3 \
    --checkpointing_steps 300 \
    --gradient_accumulation_steps 5 \
    --validation_steps 300 \
    --enable_gradient_checkpointing \
    --lr_scheduler constant \
    --report_to wandb \
    --seed 1234 \
    --mixed_precision fp16 \
    --clip_length 25 \
    --min_guidance_scale 1.0 \
    --max_guidance_scale 3.0 \
    --noise_aug_strength 0.01 \
    --bbox_dropout_prob 0.1 \
    --num_demo_samples 15 \
    --backprop_temporal_blocks_start_iter -1 \
    --num_train_epochs 30 \
    --train_H 320 \
    --train_W 512 \
    --resume_from_checkpoint latest \
    --wandb_entity $WANDB_ENTITY 