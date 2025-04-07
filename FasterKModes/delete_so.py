import os
import glob
import subprocess   
from appdirs import user_data_dir

def delete_all_generated_so():
    folder = user_data_dir("FasterKModes")
    # Get the list of .so files
    so_files = glob.glob(os.path.join(folder, "*.so"))
    
    if not so_files:
        print(f"No .so files found in {folder}.")
        return
    
    print("The following .so files will be deleted:")
    for file in so_files:
        print(" -", file)
    
    # Ask for user confirmation
    confirm = input("Do you want to delete the above files? (y/n): ")
    if confirm.lower() != 'y':
        print("Deletion cancelled.")
        return
    
    # Create and execute the deletion command
    cmd = f"rm {' '.join(so_files)}"
    result = subprocess.run(cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        print("Command execution failed:")
        print(result.stderr)
    else:
        print("Deletion complete.")
