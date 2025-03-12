import os
import shutil
import re

# Define the source directory where the files are located
source_dir = r"C:\Users\fiore\Desktop\replica_trick\replica-trick"
target_base_dir = os.path.join(source_dir, "app", "science", "analysis")

# Ensure the base target directory exists
os.makedirs(target_base_dir, exist_ok=True)

# Regex pattern to extract the version number B from filename
pattern = re.compile(r"generated_pseudodata_replica_(.+)_v(\d+)\.png")


# Iterate over files in the source directory
for filename in os.listdir(source_dir):
    match = pattern.match(filename)
    if match:
        version = match.group(1)
        target_dir = os.path.join(target_base_dir, f"version_{version}", "plots")
        
        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        # Move the file
        shutil.move(os.path.join(source_dir, filename), os.path.join(target_dir, filename))
        print(f"Moved {filename} -> {target_dir}")

print("File organization complete.")
