import os
from pathlib import Path
import argparse
from tqdm import tqdm
import math
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import ngff_utils, msi_utils
import numpy as np

from pylibCZIrw import czi as pyczi

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataPath", help="The path to your data")
parser.add_argument("--extension", help="The extension of the files to be processed", default='.czi')
parser.add_argument("--maxZSlices", help="Maximum number of Z slices per substack", type=int, default=300)

args = parser.parse_args()
print(args.dataPath)

if args.dataPath is None:
    print("Please provide a data path")
    exit(1)

basedir = Path(args.dataPath)


def prepare_savename(name, index):
    """
    Change save name to match what is expected for 3D stitching script
    
    Parameters
    ----------
    name : string
        File name to be change.
    index : int
        index of the substack.

    Returns
    ----------
    savename : string
        string file name compatible with following scripts
    """
    name = name.replace(' ', '_')
    name_firstpart = name[:name.index('-Airyscan')]
    name_lastpart = name[name.index('-Airyscan'):]
    savename = name_firstpart + '_sub' + str(index + 1) + '-Scene-1' + name_lastpart + '.zarr'
    return savename


def main(datapath='.', extension='.czi', max_z_slices=300):
    filelist = os.listdir(datapath)

    filelist = [f for f in filelist if f.find(extension) > 0]
    filelist.sort()
    print('Nr of %s files in dir: %s' % (extension, len(filelist)))

    savedir = Path( str(basedir) + '/substack_czi/')
    savedir.mkdir(parents=True, exist_ok=True)
    print('Saving output to:', savedir)

    for file in tqdm(filelist, desc='Processing files'):
        file_path = str(datapath / file)
        filename_noext = file[:file.index(extension)]
        filename_noext = filename_noext.replace(' ', '_')
        print('File path:', file_path)
        # print("File name:", filename_noext)

        with pyczi.open_czi(file_path) as czidoc:
            md_dic = czidoc.metadata
            tbd = czidoc.total_bounding_box
            pixelsize_x = float(md_dic['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value'])
            pixelsize_y = float(md_dic['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1]['Value'])
            pixelsize_z = float(md_dic['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['Value'])

        print('Image dimensions:', tbd)

        # pixel sizes need to be converted from m to microns
        scale_x = pixelsize_x / 10**-6
        scale_y = pixelsize_y / 10**-6
        scale_z = pixelsize_z / 10**-6
        
        scale = {'z': pixelsize_z, 'y': pixelsize_y, 'x': pixelsize_x}
        print('Voxel scales: %s' % scale)

        n_slices = tbd['Z'][1]

        if n_slices > max_z_slices:
            # compute  z ranges to load and save
            z_split_factor = int(math.ceil(n_slices / max_z_slices)) # number of substacks per volume to fit in memory
            stack_range_subets = [None]*z_split_factor
            z_middle = int(math.floor(n_slices / z_split_factor)) # compute substack size to fit in memory
            z_middle_overlap = int(math.ceil(z_middle * 0.05)) # computer substack 5% overlap in Z
            print("Splitting into %s substacks" % z_split_factor)

            for j in range(len(stack_range_subets)):
                if j == 0:
                    stack_range_subets[j] = (0, z_middle + z_middle_overlap)
                elif j == len(stack_range_subets)-1:
                    stack_range_subets[j] = ((j * z_middle) - z_middle_overlap, n_slices)
                else:
                    stack_range_subets[j] =  (((j * z_middle) - z_middle_overlap), ((j + 1) * z_middle) + z_middle_overlap)
            
            print("Substack ranges:", stack_range_subets)

            for i in tqdm(range(len(stack_range_subets)), desc='Processing substacks'):
                with pyczi.open_czi(file_path) as cziimg:
                    tbd = cziimg.total_bounding_box
                    im_data = np.zeros((tbd['T'][1], tbd['C'][1], stack_range_subets[i][1] - stack_range_subets[i][0], tbd['Y'][1], tbd['X'][1]))
                    print("Image shape: ", im_data.shape)

                    for t in range(tbd['T'][1]):
                        for c in range(tbd['C'][1]):
                            for z in range(stack_range_subets[i][0], stack_range_subets[i][1]):
                                temp = cziimg.read(
                                    plane = {'C': c, "T": t, "Z": z},
                                    scene = 0,
                                )
                                im_data[t, c, z - stack_range_subets[i][0]] = temp.squeeze()

                savename = prepare_savename(filename_noext, i)
                print('Saving with file name: %s' % savename)

                subset_save_path = str(savedir) + '/' + savename
                print('Subset save path:', subset_save_path)
                    
                sim = si_utils.get_sim_from_array(
                        im_data,
                        dims=["t", "c", "z", "y", "x"],
                        scale=scale,
                        )

                # write to OME-Zarr
                ngff_utils.write_sim_to_ome_zarr(sim, subset_save_path, overwrite=True)
        else:
            print('Image has less than max Z slices (%s), skipping' % max_z_slices)
            continue
        print('-------------------')
    print('====================')

if __name__ == '__main__':
    main(basedir, args.extension, args.maxZSlices)