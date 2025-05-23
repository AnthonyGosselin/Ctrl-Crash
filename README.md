# Ctrl-Crash: Controllable Diffusion for Realistic Car Crashes

## Setting Up Dependencies

### Installing Dependencies

1. Create a new conda environment:
   ```bash
   conda create -n ctrl-crash python=3.9
   conda activate ctrl-crash
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Downloading Models

- **Pretrained Models**:  
  Link to the pretrained model checkpoint: https://drive.google.com/drive/folders/1zME-pcQnW2ThZwrkZcVJV-_OhKtXMIRJ?usp=sharing

- **HuggingFace Model**:  
  For video diffusion training, the script uses the model ID `stabilityai/stable-video-diffusion-img2vid-xt`. Ensure you have the necessary permissions and credentials to download it.

## Training Scripts

The repository includes four training scripts for different scenarios:

1. **Box2Video Training (Action-Conditioned)**
   - `box2video_train_action_multigpu.sh`: Multi-GPU training for action-conditioned video generation
   - `box2video_train_action_singlegpu.sh`: Single-GPU training for action-conditioned video generation
   - These scripts train a model that generates videos conditioned on bounding boxes and action types
   - Key features: action conditioning, contiguous bbox masking, and validation on first step

2. **Video Diffusion Training**
   - `mmau_train_video_diffusion_multigpu.sh`: Multi-GPU training for video diffusion
   - `mmau_train_video_diffusion_singlegpu.sh`: Single-GPU training for video diffusion
   - These scripts train a video diffusion model based on Stable Video Diffusion
   - Key features: temporal block backpropagation and bbox dropout

### Running the Scripts

1. Configure the user-specific settings at the top of each script:
   ```bash
   DATASET_PATH="<path/to/datasets>"  # Your dataset directory
   NAME="<experiment_name>"           # Name for this training run
   OUT_DIR="<path/to/output>/${NAME}" # Where to save results
   PROJECT_NAME='<wandb_project_name>' # Weights & Biases project name
   WANDB_ENTITY='<wandb_username>'    # Your Weights & Biases username
   PRETRAINED_MODEL_PATH="<path/to/pretrained/model>" # Model checkpoint path
   ```

2. Make the script executable and run:
   ```bash
   chmod +x box2video_train_action_multigpu.sh
   ./box2video_train_action_multigpu.sh
   ```

3. Monitor training:
   - Checkpoints are saved every 300 steps
   - Training progress is logged to Weights & Biases
   - Validation samples are generated periodically

Note: The scripts use Accelerate for distributed training. Make sure you have the appropriate config files in the `config/` directory.

## Additional Tools and Documentation

- **Dataset Preprocessing**  
  Scripts and instructions for preparing datasets (frame extraction, cropping, label generation, train/val split) are provided in the [preprocess README](src/preprocess/README.md). See that file for details on supported datasets and usage examples.

- **Video Quality Evaluation**  
  Tools for evaluating generated videos using metrics such as FVD and JEDi are described in the [eval README](src/eval/README.md). This includes instructions for generating videos, running evaluation scripts, and interpreting results.

See the linked READMEs above for more information and advanced usage instructions.
