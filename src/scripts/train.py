import yaml
import os
import numpy as np 
import random
import pandas as pd
import tensorflow as tf
from data.dataset import build_dataframe,split_dataframe
from models.cnn import build_model


def load_yaml(path:str)-> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config

def set_seeds(seed:int)->None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main(config_path:str):
    confg = load_yaml(config_path)
    seed = confg["project"]["seed"]
    set_seeds(seed)
    confg_paths = confg["paths"]
    df= build_dataframe(confg_paths["dataset_dir"])
    train_df, val_df,test_df = split_dataframe(df=df)
    print(f"train: {len(train_df)} | val: {len(val_df)} | Test: {len(test_df)}")


if __name__ == "__main__":
    main("config/config.yaml")




    











