import argparse #test

from src.eval.generate_samples import generate_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test samples from MMAU dataset")
    parser.add_argument('--model_path', type=str, default="/network/scratch/a/anthony.gosselin/Results/ctrlv/box2video_apr30_actionbranch_ckpt30900_safecopy", help='Model checkpoint used for generation')
    parser.add_argument('--data_root', type=str, default="/network/scratch/a/anthony.gosselin/Datasets", help='Dataset root path')
    parser.add_argument('--output_path', type=str, default="/network/scratch/a/anthony.gosselin/CarCrashEvalSamples/default_batch", help='Video output path')
    parser.add_argument('--disable_null_model', action="store_true", default=False, help='For uncond noise preds, whether to use a null model')
    parser.add_argument('--use_factor_guidance', action="store_true", default=False, help='')
    parser.add_argument('--num_demo_samples', type=int, default=10, help='Number of samples to collect for generation')
    parser.add_argument('--max_output_vids', type=int, default=200, help='Exit program once this many videos have been generated')
    parser.add_argument('--num_gens_per_sample', type=int, default=1, help='Number videos to generate for each test case')
    parser.add_argument('--eval_output', action="store_true", default=False, help='')
    parser.add_argument('--seed', type=int, default=None, help='')
    parser.add_argument('--dataset', type=str, default="mmau")
    parser.add_argument(
        "--bbox_mask_idx_batch",
        nargs="+",
        type=int,
        default=[None],
        choices=list(range(25+1)),
        help="Where to start the masking, multiple values represent multiple different test cases for each sample",
    )
    parser.add_argument(
        "--force_action_type_batch",
        nargs="+",
        type=int,
        default=[None],
        choices=[0, 1, 2, 3, 4],
        help="Which action type to force, multiple values represent multiple different test cases for each sample",
    )
    parser.add_argument(
        "--guidance_scales",
        nargs="+",
        type=int,
        default=[(1, 9)],
        help="Guidance progression to use, multiple values represent multiple different test cases for each sample",
    )
    
    args = parser.parse_args()

    generate_samples(args)