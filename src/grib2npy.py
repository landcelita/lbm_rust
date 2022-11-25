import os
import pygrib
import numpy as np
from dotenv import load_dotenv
import glob

load_dotenv(override=True)
DATA_DIR = os.getenv("DATA_DIR")

def main():
    files = glob.glob(DATA_DIR + 'data/**/*')
    for file in files:
        basename_without_ext = os.path.splitext(os.path.basename(file))[0]
        print(basename_without_ext)
        grbs = pygrib.open(file)
        pressure = grbs.select()[0].values
        u_hori = grbs.select()[1].values
        u_vert = grbs.select()[2].values
        np.save(DATA_DIR + 'npy/u_vert_' + basename_without_ext, u_vert)
        np.save(DATA_DIR + 'npy/u_hori_' + basename_without_ext, u_hori)
        np.save(DATA_DIR + 'npy/pressure_' + basename_without_ext, pressure)

if __name__ == '__main__':
    main()