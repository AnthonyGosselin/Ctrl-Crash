# Video Dataset Processing Tools

This directory contains tools for processing and filtering video datasets. There are two main types of tools:

1. Dataset Preprocessing Tools (`preprocess_*.py`)
2. Dataset Filtering Tool (`filter_dataset_tool.py`)

## Dataset Preprocessing Tools

These scripts process raw video datasets (DADA2000, CAP, and Russia Car Crash) by:
- Extracting frames at specified FPS
- Cropping frames to desired dimensions
- Generating object detection labels
- Creating train/val splits

### Usage

Basic usage with default settings:
```bash
# For DADA2000 dataset
python preprocess_dada_dataset.py

# For CAP dataset
python preprocess_cap_dataset.py

# For Russia Car Crash dataset
python preprocess_russia_dataset.py
```

Advanced usage with custom settings:
```bash
python preprocess_dada_dataset.py \
    --dataset_root /path/to/datasets \
    --dataset_dir /path/to/raw/dataset \
    --out_directory /path/to/output \
    --out_fps 15 \
    --skip_extraction \
    --skip_labels \
    --skip_split
```

### Common Arguments
- `--dataset_root`: Root directory for datasets
- `--dataset_dir`: Directory containing the raw dataset
- `--out_directory`: Output directory (defaults to {dataset_root}/dataset_name)
- `--skip_extraction`: Skip frame extraction step
- `--skip_labels`: Skip label generation step
- `--skip_split`: Skip train/val split step

### Dataset-Specific Arguments
- DADA2000:
  - `--out_fps`: Output frames per second (default: 12)
- CAP:
  - `--reverse`: Process samples in reverse order
- Russia:
  - `--process_train`: Process training set (default is validation set only)

## Dataset Filtering Tool

A tool for manually reviewing and filtering video datasets. It provides an interactive interface to review video frames and mark them as high quality or rejected. The tool can also automatically detect upscaled videos and scene changes to help with the filtering process.

### Features
- Interactive video frame review with keyboard controls
- Automatic detection of upscaled videos
- Scene change detection
- Caching support for faster processing
- Support for both single-category and multi-category datasets

### Usage

Basic usage with default settings:
```bash
python filter_dataset_tool.py --dataset_name my_dataset
```

Advanced usage with all features enabled:
```bash
python filter_dataset_tool.py \
    --dataset_name my_dataset \
    --start_idx 0 \
    --data_dir ./custom/path/to/images \
    --output_root ./custom/output/path \
    --use_cache
```

### Keyboard Controls
- `w`: Next frame
- `s`: Previous frame
- `d`: Next video
- `a`: Previous video
- `r`: Reject video
- `h`: Mark as high quality
- `p`: Increase playback speed
- `l`: Decrease playback speed
- `ESC`: Exit

### Command Line Arguments
- `--dataset_name`: Name of the dataset directory (required)
- `--start_idx`: Starting index for video review (default: 0)
- `--data_dir`: Custom data directory path (default: ./{dataset_name}/images)
- `--output_root`: Custom output root directory (default: ./{dataset_name})
- `--disable_sort_by_upsample`: Disable sorting by upsampling factor
- `--disable_check_scene_changes`: Disable scene change detection
- `--single_category`: Process videos from a single category directory
- `--use_cache`: Use cache to speed up processing 