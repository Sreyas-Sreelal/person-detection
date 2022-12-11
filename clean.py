import os
import glob
import shutil

def clean_dir(dir):
    files = glob.glob(dir+'/*')
    for f in files:
        try:
            os.remove(f)
        except PermissionError:
            shutil.rmtree(f)
clean_dir('model')
clean_dir('processed')
