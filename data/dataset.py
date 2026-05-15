import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
def build_dataframe(dataset_dir: str) -> pd.DataFrame :
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
    return df 



def split_dataframe(
    df:pd.DataFrame, 
    val_split:float = 0.15, 
    test_split: float = 0.15, 
    seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_df,temp_df = train_test_split(df,test_size=(val_split + test_split),random_state=seed,stratify=df['label'])

    val_df,test_df = train_test_split(temp_df,test_size=(test_split / (val_split + test_split)),random_state=seed,stratify=temp_df['label'])

    return train_df,val_df,test_df



