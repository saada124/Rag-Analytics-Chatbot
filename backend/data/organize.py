import os
from pathlib import Path
import shutil

def organize_folder(target_directory):
    target_path = Path(target_directory)
    
    if not target_path.exists():
        print(f"Error: The directory '{target_directory}' does not exist.")
        return

    print(f"Sorting files in: {target_path.resolve()}\n" + "-"*40)

    for item in target_path.iterdir():
        
        if item.is_file():
            file_extension = item.suffix.lower().replace('.', '')
            if not file_extension:
                file_extension = "no_extension"

            folder_name = f"{file_extension.upper()} Files"
            destination_folder = target_path / folder_name
            destination_folder.mkdir(exist_ok=True)
            destination_file_path = destination_folder / item.name

            #move the file
            try:
                shutil.move(str(item), str(destination_file_path))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

    print("-"*40 + "\nSorting complete!")

if __name__ == "__main__":
    FOLDER_TO_SORT = "./data" 
    organize_folder(FOLDER_TO_SORT)