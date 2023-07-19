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
parser.add_argument("--train_data", type=str, help="path to train data")
parser.add_argument("--test_data", type=str, help="path to test data")
args = parser.parse_args()

# Start Logging
mlflow.start_run()

print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

#print("input data:", args.data)

arr = os.listdir(args.data)
print(arr)

########################################
df = []
for filename in arr:
    print("reading file: %s ..." % filename)
    with open(os.path.join(args.data, filename), "r") as handle:
        # print (handle.read())
        # ('input_df_%s' % filename) = pd.read_csv((Path(args.training_data) / filename))
        input_df = pd.read_csv((Path(args.data) / filename))
        df.append(input_df)
        
print(df)

##########################################

# df = pd.read_csv(args.data)
# print(df)

train_df, test_df = train_test_split(df,test_size=0.3,random_state=4)

# output paths are mounted as folder, therefore, we are adding a filename to the path
train_df = train_df.to_csv((Path(args.train_data) / "train_data.csv"))

test_df = test_df.to_csv((Path(args.test_data) / "test_data.csv"))

# Stop Logging
mlflow.end_run()

