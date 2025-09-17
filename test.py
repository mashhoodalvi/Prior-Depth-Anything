import torch
import time
from prior_depth_anything import PriorDepthAnything
import argparse
import logging


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Prior depth anything."
    )
    parser.add_argument(
        "--input_rgb",
        type=str,
        help="Path to input image",
    )

    parser.add_argument(
        "--prior_depth",
        type=str,
        required=True,
        help="Path to prior depth",
    )

    args = parser.parse_args()


    image_path = args.input_rgb
    prior_path = args.prior_depth

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    priorda = PriorDepthAnything(device=device)
    a = time.perf_counter()
    output = priorda.infer_one_sample(image=image_path, prior=prior_path, visualize=True)
    b = time.perf_counter()
    print(b-a)