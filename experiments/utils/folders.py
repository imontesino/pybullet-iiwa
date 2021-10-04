"""Utils for folder and file management"""
import os
import shutil
from typing import List

def delete_contents(folder):
    """Remove all contents and subdirectories in a given directory

    :param folder: the path of the directory to delete
    """
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

def choose_model_file() -> str:
    """Chooose a model file from a list of files in the models save directory"""
    items = os.listdir("models")

    model_files: List[str] = []
    for name in items:
        if name.endswith(".zip"):
            model_files.append(name)

    print("----- Model Files ------")
    for i, file in enumerate(model_files):
       print(str(i)+": "+file)

    while True:
        user_option = int(input("Model to load: "))
        if user_option < len(model_files):
            break
        else:
            print("Wrong option")

    return "models/"+model_files[user_option].replace(".zip","")
