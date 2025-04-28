import os
import shutil

path = "results"

if os.path.isfile(path):
    os.remove(path)
    print(f"Datei '{path}' wurde gelöscht.")
elif os.path.isdir(path):
    shutil.rmtree(path)
    print(f"Verzeichnis '{path}' wurde gelöscht.")
else:
    print(f"'{path}' ist weder eine Datei noch ein Verzeichnis.")
