# process_data.py

import os
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split

input_dir = "data/Brain_dataset"  
output_dir = "data/Hippocampe_dataset_mci_only_test"
label_file = "data/list_standardized_tongtong_2017.csv"
output_csv = os.path.join(output_dir, "hippocampi_labels.csv")

if os.path.exists(output_csv): # check if output CSV already exists to avoid redundant processing
    print(f"Le fichier {output_csv} existe déjà.")
    exit(0)

hypocampe1_coords = (slice(40, 80), slice(90, 130), slice(40, 80)) # define hippocampal region coordinates 
hypocampe2_coords = (slice(100, 140), slice(90, 130), slice(40, 80)) # define hippocampal region coordinates 

train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")
for subdir in [train_dir, val_dir, test_dir]:
    os.makedirs(subdir, exist_ok=True)

metadata = pd.read_csv(label_file)
metadata.columns = [
    "Subject ID", "Rooster ID", "Age", "Sex", "Group",
    "Conversion", "MMSE", "RAVLT", "FAQ", "CDR-SB", "ADAS11"
]
metadata['Group'] = metadata['Group'].replace({'EMCI': 'MCI', 'LMCI': 'MCI'})

metadata = metadata[~((metadata['Group'] == 'MCI') & (metadata['Conversion'] == 5))]

def map_labels_cnn(row):
    if row['Group'] == 'CN':
        return 0  # CN -> 0
    elif row['Group'] == 'AD':
        return 1  # AD -> 1
    elif row['Group'] == 'MCI':
        if row['Conversion'] == 4: # Stable MCI -> 0, others -> 1
            return 0
        else:
            return 1
    else:
        return None

metadata['labels_cnn'] = metadata.apply(map_labels_cnn, axis=1)

def get_subject_id_from_filename(file_name):
    import re
    match = re.search(r'\d{3}_S_\d{4}', file_name) # search for the pattern matching a subject ID (e.g., '123_S_4567')

    return match.group() if match else None

def extract_hippocampes(file_path, output_dir):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        
        hippocampus1 = data[hypocampe1_coords] # extract the first hippocampal regions using predefined coordinates
        hippocampus2 = data[hypocampe2_coords] # extract the second hippocampal regions using predefined coordinates
        
        base_name = os.path.basename(file_path).replace(".nii.gz", "")
        hippocampus1_path = os.path.join(output_dir, f"{base_name}_hippocampus1.nii.gz")
        hippocampus2_path = os.path.join(output_dir, f"{base_name}_hippocampus2.nii.gz")
        
        nib.save(nib.Nifti1Image(hippocampus1, None), hippocampus1_path)
        nib.save(nib.Nifti1Image(hippocampus2, None), hippocampus2_path)
        
        return {'Hippocampus1': hippocampus1_path, 'Hippocampus2': hippocampus2_path}
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_and_save(files, category_dir, metadata):
    records = []
    for file_name in files:
        subject_id = get_subject_id_from_filename(file_name) # extract subject ID from the filename
        file_path = os.path.join(input_dir, file_name) # construct the full file path
        if subject_id and os.path.exists(file_path): # check if subject ID exists and file path is valid
            paths = extract_hippocampes(file_path, category_dir)
            if paths: # continue only if extraction was successful
                row = metadata[metadata["Subject ID"] == subject_id]
                if not row.empty: # ensure there is a match in the metadata
                    group = row.iloc[0]["Group"]
                    label_cnn = row.iloc[0]["labels_cnn"]
                    for hip_name, hip_path in paths.items():
                        records.append({
                            "File": hip_path,
                            "Subject ID": subject_id,
                            "Hippocampus": hip_name,
                            "Group": group,
                            "labels_cnn": label_cnn
                        })
    return records

file_list = [
    f for f in os.listdir(input_dir) 
    if f.endswith(".nii.gz") and "mask" not in f
]

file_df = pd.DataFrame({
    "FileName": file_list,
    "SubjectID": [get_subject_id_from_filename(f) for f in file_list]
})

merged_df = file_df.merge(metadata, left_on="SubjectID", right_on="Subject ID", how="inner")

ad_cn_df = merged_df[merged_df["Group"].isin(["AD", "CN"])] # AD and CN groups
mci_df = merged_df[merged_df["Group"] == "MCI"] # MCI group

train_df, val_df = train_test_split(
    ad_cn_df, test_size=0.2, stratify=ad_cn_df["labels_cnn"], random_state=42
)

train_records = process_and_save(train_df["FileName"], train_dir, metadata)
val_records = process_and_save(val_df["FileName"], val_dir, metadata)
test_records = process_and_save(mci_df["FileName"], test_dir, metadata)

all_records = train_records + val_records + test_records
labels_df = pd.DataFrame(all_records)
labels_csv_path = os.path.join(output_dir, "hippocampi_labels.csv")
labels_df.to_csv(labels_csv_path, index=False)

print(f"Train records: {len(train_records)}")
print(f"Validation records: {len(val_records)}")
print(f"Test records: {len(test_records)}")
print(f"Labels updated and saved in: {labels_csv_path}")
