import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf


def build_dataframe(dataset_dir: str) -> tuple[pd.DataFrame,dict]:
    extensions = {".png",".jpeg",".jpg"}
    path = Path(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(f"Le dossier {dataset_dir} n'existe pas ")
    if not path.is_dir():
        raise NotADirectoryError(f"{dataset_dir} n'est pas un dossier ")
    
    records = []
    
    for class_dir in path.iterdir() :
        if  not class_dir.is_dir():
            continue
        label = class_dir.name
        for image_path in class_dir.iterdir():
            if (image_path.suffix.lower() in extensions and image_path.is_file()) :
                records.append({

                    "filepath": str(image_path.resolve()), 
                    "label" : label

                }
               
                )
    if not records:
        raise ValueError(f"Aucune image trouvée dans {dataset_dir}")

    df = pd.DataFrame(records,columns=["filepath", "label"])

    classes = sorted(df["label"].unique())

    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    df["label_int"] = df["label"].map(class_to_index)

    return df , class_to_index



def split_dataframe(
    df:pd.DataFrame, 
    val_split:float = 0.15, 
    test_split: float = 0.15, 
    seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_df,temp_df = train_test_split(df,test_size=(val_split + test_split),random_state=seed,stratify=df['label'])

    val_df,test_df = train_test_split(temp_df,test_size=(test_split / (val_split + test_split)),random_state=seed,stratify=temp_df['label'])

    return train_df,val_df,test_df



def build_tf_dataset(df:pd.DataFrame, batch_size:int,shuffle=False)-> tf.data.Dataset:
    ds = tf.data.Dataset.from_slices(
        (
            df["filepath"].values,
            df["label_int"].values
        )
    )

    def load_image(filepath,label):
        image = tf.io.read_file(filepath)

        image = tf.image.decode_png(image,channels=1)

        image = tf.image.convert_dtype(image, tf.float32)

        return image, label
    
    ds = ds.map(load_image,num_parallel_calls = tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))

    ds = ds.batch(batch_size)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
