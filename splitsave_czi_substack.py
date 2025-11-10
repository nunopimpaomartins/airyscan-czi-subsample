import os
from pathlib import Path
import argparse
from tqdm import tqdm
import math

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

def main(datapath='.', extension='.czi', max_z_slices=300):
    filelist = os.listdir(datapath)

    filelist = [f for f in filelist if f.find(extension) > 0]
    filelist.sort()
    print('Nr of czi files in dir:', len(filelist))

    savedir = Path(str(basedir) + '/substack_czi/')
    savedir.mkdir(parents=True, exist_ok=True)
    print('Saving output to:', savedir)

    for file in tqdm(filelist, desc='Processing files'):
        filepath = datapath / file
        filename_noext = file[:file.index(extension)]
        filename_noext = filename_noext.replace(' ', '_')
        print('Processing file:', filename_noext)

        with pyczi.open_czi(filepath) as czidoc:
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

        n_slices = tbd['Z'][0]

        if n_slices > max_z_slices:
            # compute  z ranges to load and save
            z_split_factor = int(math.ceil(n_slices / max_z_slices)) # number of substacks per volume to fit in memory
            stack_range_subets = [None]*z_split_factor
            z_middle = int(math.floor(n_slices / z_split_factor)) # compute substack size to fit in memory
            z_middle_overlap = int(math.ceil(z_middle * 0.05)) # computer substack 5% overlap in Z

            for j in range(0, len(stack_range_subets)):
                if j == 0:
                    stack_range_subets[j] = (1, ((j + 1) * z_middle) + z_middle_overlap)
                elif j == len(stack_range_subets)-1:
                    stack_range_subets[j] = ((j * z_middle) - z_middle_overlap , n_slices)
                else:
                    stack_range_subets[j] =  (((j * z_middle) - z_middle_overlap), ((j + 1) * z_middle) + z_middle_overlap)

            img_data = img.get_image_dask_data(img.dims.order[1:], M=tile) #TODO better strategy to exclude dimension to split
            img_data_tile = img_data.compute()

            ch_names = {}
            for i in range(len(img.channel_names)):
                ch_names[i] = img.channel_names[i]

            tile_save_path = str(savedir) + '/' + filename_noext + '_tile' + str(tile+1).zfill(2) + '.czi'
            print('Tile save path:', tile_save_path)

            with pyczi.create_czi(tile_save_path, exist_ok=True) as czidoc_w:
                for t in range(img.dims['T'][0]):
                    for c in range(img.dims['C'][0]):
                        for z in range(img.dims['Z'][0]):
                            temp_image = img_data_tile[t][c][z]
                            czidoc_w.write(
                                data=temp_image,
                                plane={
                                    'T': t,
                                    'C': c,
                                    'Z': z,
                                },
                                compression_options = "zstd0:ExplicitLevel=2"
                            )
                
                czidoc_w.write_metadata(
                    filename_noext + '_tile' + str(tile+1).zfill(2),
                    channel_names=ch_names,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    scale_z=scale_z,
                )
        else:
            print('Image has less than max Z slices (%s), skipping' % max_z_slices)
            continue

                

if __name__ == '__main__':
    main(basedir, args.extension, args.maxZSlices)