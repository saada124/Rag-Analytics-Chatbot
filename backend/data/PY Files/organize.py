import os
from pathlib import Path
import shutil

def organize_folder(target_directory):
    # Convert string path to a Path object
    target_path = Path(target_directory)
    
    # Ensure the directory actually exists
    if not target_path.exists():
        print(f"Error: The directory '{target_directory}' does not exist.")
        return

    print(f"Sorting files in: {target_path.resolve()}\n" + "-"*40)

    # Iterate through all items in the directory
    for item in target_path.iterdir():
        # Skip directories, we only want to move files
        if item.is_file():
            # Get the file extension (e.g., '.csv', '.pdf') and remove the dot, uppercase it
            file_extension = item.suffix.lower().replace('.', '')
            
            # If the file has no extension, put it in an 'unknown' folder
            if not file_extension:
                file_extension = "no_extension"
                
            # Define the target folder name (e.g., "CSV Files", "PDF Files")
            folder_name = f"{file_extension.upper()} Files"
            destination_folder = target_path / folder_name

            # Create the destination folder if it doesn't exist yet
            destination_folder.mkdir(exist_ok=True)

            # Define the final destination path for the file
            destination_file_path = destination_folder / item.name

            try:
                # Move the file
                shutil.move(str(item), str(destination_file_path))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

    print("-"*40 + "\nSorting complete!")

if __name__ == "__main__":
    # Replace this with the path to the folder you want to clean up
    # Use r"C:\Users\YourName\Downloads" style format for Windows paths
    FOLDER_TO_SORT = "./data" 
    
    organize_folder(FOLDER_TO_SORT)