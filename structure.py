import os

# Define the folder structure
folder_structure = {
    ".vscode": ["settings.json"],
    ".github/workflows": ["unittests.yml"],
    "src": ["__init__.py"],
    "notebooks": ["__init__.py", "README.md"],
    "tests": ["__init__.py"],
    "scripts": ["__init__.py", "README.md"]
}

# Create the folder structure
def create_structure(base_path="."):
    for folder, files in folder_structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, "w") as f:
                f.write("")  # Create an empty file
    
    # Create standalone files
    standalone_files = [".gitignore", "requirements.txt", "README.md"]
    for file in standalone_files:
        file_path = os.path.join(base_path, file)
        with open(file_path, "w") as f:
            f.write("")  # Create an empty file

if __name__ == "__main__":
    create_structure()
    print("Folder structure created successfully!")