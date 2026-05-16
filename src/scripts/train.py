import yaml
import os
import numpy as np 
import random
import pandas as pd
import tensorflow as tf
from data.dataset import build_dataframe,split_dataframe,build_tf_dataset
from models.cnn import build_model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard,CSVLogger


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

    df,class_to_index= build_dataframe(confg_paths["dataset_dir"])

    train_df, val_df,test_df = split_dataframe(df=df)

    print(f"train: {len(train_df)} | val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = build_tf_dataset(train_df,confg['data']['batch_size'],shuffle=True)

    val_ds = build_tf_dataset(val_df,confg['data']['batch_size'],shuffle = False)

    test_ds = build_tf_dataset(test_df,confg['data']['batch_size'],shuffle=False)

    model = build_model(confg)
    model.summary()

    train_confg = confg['training']

    early_stop = EarlyStopping(
        monitor=train_confg['early_stopping']['monitor'],
        patience = train_confg['early_stopping']['patience'],
        restore_best_weights=True,
        verbose=1
    )


    checkpoint = ModelCheckpoint(
        f"{confg['paths']['model_dir']}/{confg['project']['run_id']}_best_model.keras",
        monitor=train_confg['early_stopping']['monitor'],
        save_best_only=True,
        verbose=1

    )

    reduce_lr = ReduceLROnPlateau(
        monitor=train_confg['reduce_lr']['monitor'],
        factor =train_confg['reduce_lr']['factor'],
        patience=train_confg['reduce_lr']['patience'],
        min_lr=train_confg['reduce_lr']['min_lr'],
        verbose=1
    )

    tb = TensorBoard(log_dir=confg['paths']['log_dir'])

    csv_logger = CSVLogger(f"{confg['paths']['log_dir']}/{confg['project']['run_id']}_train_log.csv",append=False)


    callbacks_list = [early_stop, checkpoint, reduce_lr, tb, csv_logger]



    history = model.fit(
        train_ds,
        epochs=train_confg['epochs'],
        validation_data=val_ds,
        callbacks=callbacks_list

    )








if __name__ == "__main__":
    main("config/config.yaml")




    











