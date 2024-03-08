from pathlib import Path

import pandas as pd
from pycocotools.coco import COCO


def split_by_coco_files(coco_path_prefix):
    split_rules_name = Path(coco_path_prefix).name
    # Accumulate patches and targets (to later build a dataframe)
    split_rows = []
    for target in ["train", "val", "test"]:
        coco = COCO(f"{coco_path_prefix}_{target}.json")

        for patch_id, patch_info in coco.imgs.items():
            # Perform a rename to match the netcdf file structure
            # e.g. 'netcdf/2019_31TCG_patch_13_11.nc' -> '2019/31TCG/2019_31TCG_patch_13_11.nc'
            patch_filename = patch_info['file_name'].removeprefix("netcdf/")
            year, tile, _, patch_x, patch_y = patch_filename.split('_')
            patch_path = Path(str(year)) / tile / patch_filename

            split_rows.append((patch_path, target))
    split_df = pd.DataFrame.from_records(split_rows, columns=['path', 'target'])
    return split_df, split_rules_name
