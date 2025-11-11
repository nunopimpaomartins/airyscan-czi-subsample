# airyscan-czi-subsample
Repository with python scripts and notebooks to processes Airyscan data too large for processing in a limited RAM workstation. 

These scripts and pipelines are intended for pre-processing Airyscan raw data which is too large and cannot be processed as a single Z stack. They have been prepared for mosaic datasets, Zeiss Zen will process each tile independenlty.

The processing bottleneck is in the number of Z planes that can be processes. 
From my experience, it requires ~70 GB of RAM / 100 z planes of images with shape `(4098 x 4098)`. For a workstation with 256 GB of RAM, ~ 300 z planes can be processed at a time. In workstations with less RAM, one has to split Z stacks into smaller subsets.

The current workflow consists of splitting raw Airyscan data into substacks for Airyscan processing in Zeiss Zen, splitting processed data into the individual tiles, stitch then in 3D, then perform the final 2D/mosaic stitching for the complete dataset.
Stitching is done with the [`multiview stitcher`](https://github.com/multiview-stitcher/multiview-stitcher/tree/main) and using OME Zarr as the data format.

### Pipeline
![Schematic Pipeline](/media/schematic_processing_pipeline.png)


## Environment

## Usage
1. Process RAW data in `Zeiss Zen` with the Zen compatible script
    - using: `Airyscan_subset_split_data.czmac`
1. This will generate multiple raw substacks. Processed them individually or in Batch in `Zeiss Zen`
1. In `conda/mamba`, activate the corresponding environment and process the following scripts.
1. Once processed, split processed files into individual tiles
    - using: `splitsave_czi_tile.py`
1. Use the script to stitch tiles in 3D
    - using: `1_multiview_stitcher_3d.py`
1. Stitch files in 2D using files generated from previous script
    -using:" `2_multiview_stitcher_2d.py`


### Airyscan_subset_split_data.czmac


### splitsave_czi_tile.py


### 1_multiview_stitcher_3d.py


### 2_multiview_stitcher_2d.py


## TO DO
- remove time (T) dimension from loading data in scripts for stitching. Image data should not have time dimension, if it has, probably it will break or overwrite data.