import os
from pathlib import Path
import argparse
from tqdm import tqdm

import dask.array as da 
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import ngff_utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataPath", help="The path to your data")
parser.add_argument("--extension", help="The extension of the files to be processed", default='.czi')

args = parser.parse_args()

# get Source path
if args.dataPath is None:
    print("Please provide a data path")
    exit(1)
basedir = Path(args.dataPath)

# get the image reader based on the file extension
if args.extension == '.czi':
    from bioio import BioImage
    import bioio_czi
else:
    from tifffile import imread


def main(datapath='.', extension='.czi'):
    print('Processing folder: %s' % datapath)
    filelist = os.listdir(datapath)

    filelist = [f for f in filelist if f.endswith(extension)]
    filelist.sort()
    print('Nr of %s files in dir: %i' % (extension, len(filelist)))

    savedir = Path(str(datapath) + '/converted_files/')
    savedir.mkdir(parents=True, exist_ok=True)
    print('Saving output to: %s' % savedir)

    for file in tqdm(filelist):

        # Getting image data voxel scales
        file_path = str(datapath / file)
        img = BioImage(
            file_path,
            reader=bioio_czi.Reader,
            reconstruct_mosaic=False,
            include_subblock_metadata=True,
            use_aicspylibczi=True
        )
        scale = {'z': img.scale.Z, 'y': img.scale.Y, 'x': img.scale.X}
        print('Voxel scales: %s' % scale)
        
        overwrite = True
        # remove spaces from filename
        file_savename = file[:file.index(extension)].replace(' ', '_') + '.zarr'
        print('Saving to OME-Zarr file as: %s' % file_savename)

        zarr_path = os.path.join(savedir, file_savename)
        
        # get data dimensions without T axis from metadata
        im_data = img.get_image_data(img.dims.order[img.dims.order.index('T')+1:])

        sim = si_utils.get_sim_from_array(
            im_data,
            dims=["c", "z", "y", "x"],
            scale=scale,
            )

        # write to OME-Zarr
        ngff_utils.write_sim_to_ome_zarr(sim, zarr_path, overwrite=overwrite)
        
        print('====================')
    print('Done!')


if __name__ == '__main__':
    main(datapath=basedir, extension=args.extension)