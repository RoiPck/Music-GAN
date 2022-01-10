import os
import shutil
import pandas as pd

if __name__ == "__main__":
    # Constant variables
    ROOT = os.path.join("data", "sound_files")
    DST_FOLDER = os.path.join("data", "prepared")
    DST_SOUND_FOLDER = os.path.join(DST_FOLDER, "sounds")
    COLUMNS = ["Path", "Type str", "Type int", "Old Name"]

    # Variables initialisation
    labeled_data = []
    dir_number = 0

    # Removing already existing soundfiles
    for files in os.listdir(DST_SOUND_FOLDER):
        path = os.path.join(DST_SOUND_FOLDER, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

    # Counting number of files to be labelled and copied
    n_files = sum([len(files) for root, dirs, files in os.walk(ROOT)])
    print(f"Copying and labeling {n_files} files ...")

    # Copying and labeling all files
    for root, dirs, files in os.walk(ROOT):
        file_number = 0
        for f in files:
            d = os.path.basename(root)
            new_name = str(file_number + 1000 * dir_number) + ".wav"
            labeled_data.append([new_name, d, dir_number, f])
            dst = os.path.join(DST_SOUND_FOLDER, new_name)
            src = os.path.abspath(os.path.join(ROOT, d, f))
            shutil.copy(src, dst)
            file_number += 1
        dir_number += 1

    # Creating the csv file
    df = pd.DataFrame(labeled_data, columns=COLUMNS)
    df.to_csv(os.path.join(DST_FOLDER, "sound_data.csv"), index=False)


    print("Dataset successfully created !")