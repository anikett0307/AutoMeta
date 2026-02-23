import os
import shutil
import pandas as pd

# Paths based on your folder structure
csv_path = "../data/HAM10000/HAM10000_metadata.csv"
image_folder = "../data/HAM10000/all_images"
output_folder = "../data/HAM10000"

# Mapping from 'dx' to folder names
lesion_map = {
    "mel": "Melanoma",
    "nv": "Nevus",
    "bcc": "Basal_Cell_Carcinoma",
    "akiec": "Actinic_Keratosis",
    "df": "Dermatofibroma",
    "vasc": "Vascular_Lesions",
    "bkl": "Benign_Keratosis"
}

# Load metadata
df = pd.read_csv(csv_path)

# Move each image into the correct folder
for idx, row in df.iterrows():
    img_filename = row['image_id'] + ".jpg"
    lesion_type = row['dx']
    source = os.path.join(image_folder, img_filename)
    
    if lesion_type in lesion_map:
        target_folder = os.path.join(output_folder, lesion_map[lesion_type])
        destination = os.path.join(target_folder, img_filename)
        
        if os.path.exists(source):
            shutil.move(source, destination)
        else:
            print(f"⚠ File not found: {img_filename}")
    else:
        print(f"❌ Unknown lesion type: {lesion_type}")

print("✅ Dataset organization complete.")