import os
from pathlib import Path
import argparse
from tqdm import tqdm

import xarray as xr
import numpy as np
import dask.diagnostics
import dask.array as da 
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import (
    fusion,
    io,
    msi_utils,
    vis_utils,
    ngff_utils,
    param_utils,
    registration
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataPath", help="The path to your data")
parser.add_argument("--extension", help="The extension of the files to be processed", default='.czi')

args = parser.parse_args()
print(args.dataPath)

if args.dataPath is None:
    print("Please provide a data path")
    exit(1)

basedir = Path(args.dataPath)
if args.extension is '.czi':
    from bioio import BioImage
    import bioio_czi
else:
    from tifffile import imread

def get_unique_names(array, substring='.'):
    """
    Get unique names from a string array
    """
    unique_names = [f[:f.index(substring)] for f in array]
    unique_names = list(set(unique_names))
    return unique_names.sort()

def main(datapath='.', extension='.czi'):
    filelist = os.listdir(datapath)

    filelist = [f for f in filelist if f.endswith(extension)]
    filelist.sort()
    print('Nr of czi files in dir:', len(filelist))

    savedir = Path(str(basedir) + '/stitched_tile_3d/')
    savedir.mkdir(parents=True, exist_ok=True)
    print('Saving output to:', savedir)

    original_filenames = get_unique_names(filelist, substring='_sub')
    print("Nb of unique file names:", len(original_filenames))

    for original_name in original_filenames:
        filelist_filtered = []
        for name in filelist:
            if name.find(original_name) >= 0:
                filelist_filtered.append(name)

        filelist_substacks = get_unique_names(filelist_filtered, substring='-Scene')
        n_positions = len(filelist_filtered) / len(filelist_substacks)
        n_substacks = len(filelist_substacks)

        for i in range(n_positions):
            substack_file_indexes = []
            for file in filelist:
                if file.find(original_name) >= 0 and file.endswith('_tile' + str(i + 1).zfill(2) + extension):
                    substack_file_indexes.append(filelist.index(file))

            



main(datapath=basedir, extension=args.extension)