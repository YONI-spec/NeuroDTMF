import pandas as pd
from pathlib import Path
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

