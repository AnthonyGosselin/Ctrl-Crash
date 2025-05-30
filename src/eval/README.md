# Video Quality Evaluation Tools

This directory contains scripts for evaluating video quality metrics between generated and ground truth videos. There are four main evaluation scripts:

1. `video_quality_metrics_fvd_pair.py`: Evaluates FVD (Fréchet Video Distance) between paired generated and ground truth videos
2. `video_quality_metrics_fvd_gt_rand.py`: Evaluates FVD using pre-computed ground truth statistics
3. `video_quality_metrics_jedi_pair.py`: Evaluates JEDi metric between paired generated and ground truth videos
4. `video_quality_metrics_jedi_gt_rand.py`: Evaluates JEDi metric using random ground truth samples

## Video Generation

Before running the evaluation scripts, you'll need to generate video samples using the `run_gen_videos.py` script:

```bash
python run_gen_videos.py \
    --model_path /path/to/model/checkpoint \
    --output_path /path/to/output/videos \
    --num_demo_samples 10 \
    --max_output_vids 200 \
    --num_gens_per_sample 1 \
    --eval_output
```

### Key Generation Arguments

```bash
--model_path PATH     # Path to model checkpoint (required)
--data_root PATH     # Dataset root path
--output_path PATH   # Where to save generated videos
--num_demo_samples N # Number of samples to collect for generation
--max_output_vids N  # Maximum number of videos to generate
--num_gens_per_sample N # Videos to generate per test case

# Optional arguments for controlling generation
--bbox_mask_idx_batch N1 N2 ...  # Where to start masking (0-25)
--force_action_type_batch N1 N2 ... # Force specific action types (0-4)
--guidance_scales N1 N2 ...      # Guidance scales to use
--seed N                        # Random seed for reproducibility
--disable_null_model           # Disable null model for unconditional noise
--use_factor_guidance         # Use factor guidance during generation
--eval_output                 # Enable evaluation output
```

### Action Types
- 0: Normal driving
- 1-4: Different types of crash scenarios

## Common Arguments for Evaluation

All evaluation scripts share some common command line arguments:

```bash
--vid_root PATH       # Root directory containing generated videos (required)
--samples N           # Number of samples to evaluate (default: 200)
--num_frames N        # Number of frames per video (default: 25)
--downsample_int N    # Downsample interval for frames (default: 1)
--action_type N       # Action type to filter videos (0: normal, 1-4: crash types)
--shuffle            # Shuffle videos before evaluation
```

## FVD Evaluation

### Paired Evaluation
```bash
python video_quality_metrics_fvd_pair.py \
    --vid_root /path/to/videos \
    --samples 200 \
    --num_frames 25 \
    --downsample
```

### Ground Truth Statistics Evaluation
```bash
# First, collect ground truth statistics
python video_quality_metrics_fvd_gt_rand.py \
    --vid_root /path/to/videos \
    --collect_stats \
    --samples 500 \
    --action_type 1

# Then evaluate using the collected statistics
python video_quality_metrics_fvd_gt_rand.py \
    --vid_root /path/to/videos \
    --gt_stats /path/to/stats.npz \
    --samples 200 \
    --shuffle
```

## JEDi Evaluation

### Paired Evaluation
```bash
python video_quality_metrics_jedi_pair.py \
    --vid_root /path/to/videos \
    --samples 200 \
    --num_frames 25 \
    --test_feature_path /path/to/features
```

### Ground Truth Random Evaluation
```bash
python video_quality_metrics_jedi_gt_rand.py \
    --vid_root /path/to/videos \
    --samples 200 \
    --gt_samples 500 \
    --test_feature_path /path/to/features \
    --action_type 1 \
    --shuffle
```

## Additional Notes

- The `--action_type` argument can be used to filter videos by category:
  - 0: Normal driving videos
  - 1-4: Different types of crash videos
- For FVD evaluation with ground truth statistics, you can collect statistics once and reuse them for multiple evaluations
- The JEDi metric requires a test feature path for model loading
- All scripts support shuffling of videos before evaluation for more robust results
- The default resolution for videos is 320x512 pixels 
