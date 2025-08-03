import os
import zipfile
from datetime import datetime

# Step 1: Set the current path to the path of this script
current_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_path)

# Step 2: Create a folder called "history" if it does not exist
history_path = os.path.join(current_path, 'history')
if not os.path.exists(history_path):
    os.makedirs(history_path)

# Helper function to calculate the total size of files within a directory
def get_dir_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size

# Function to decide whether a file or directory should be included
def should_include(item):
    item_path = os.path.join(current_path, item)
    if item in ['history', os.path.basename(__file__)]:
        return False
    if os.path.isdir(item_path):
        return get_dir_size(item_path) <= 30 * 1024 * 1024  # 30MB in bytes
    else:
        return os.path.getsize(item_path) <= 30 * 1024 * 1024  # 30MB in bytes

selected_items = [item for item in os.listdir(current_path) if should_include(item)]

# Step 4: Make a zip file of all selected file and folders, save it under /history
# The zip file name is in the format of "yy-mm-dd hh:mm:ss"
current_time = datetime.now().strftime("%y-%m-%d %H-%M-%S")
zip_file_name = f"{current_time}.zip"
zip_file_path = os.path.join(history_path, zip_file_name)
with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for item in selected_items:
        item_path = os.path.join(current_path, item)
        if os.path.isdir(item_path):
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, current_path)
                    zipf.write(file_path, arcname)
        else:
            arcname = os.path.relpath(item_path, current_path)
            zipf.write(item_path, arcname)

print(f"Backup completed successfully. File saved as {zip_file_name}.")
