import os
import argparse

import logging
import mlflow
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


# input and output arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to input data")
parser.add_argument("--test_train_ratio", type=float, required=False, default=0.2)
parser.add_argument("--train_data", type=str, help="path to train data")
parser.add_argument("--test_data", type=str, help="path to test data")
args = parser.parse_args()

# Start Logging
mlflow.start_run()

print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

print("input data:", args.data)

df = pd.read_csv(args.data)

train_df, test_df = train_test_split(
    df,
    test_size=args.test_train_ratio,
    random_state=42
)

# output paths are mounted as folder, therefore, we are adding a filename to the path
train_df = train_df.to_csv((Path(args.train_data) / "train_data.csv"))

test_df = test_df.to_csv((Path(args.test_data) / "test_data.csv"))

# Stop Logging
mlflow.end_run()

