import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.utils.utils_generate_samples import generate_samples_dataframe, create_train_val_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Generate sintetic dataset with train/val/test split."
    )
    parser.add_argument("output_file", help="Path to file with all combinations of samples parameters with split.")
    args = parser.parse_args()
    
    ### Generate 
    output_file = args.output_file
    
    df = generate_samples_dataframe()
    df = create_train_val_test_split(df, seed=41)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()

