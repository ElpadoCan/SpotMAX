import os

from cellacdc.data import _Data
from cellacdc import load

from . import data_path

class _SpotMaxData(_Data):
    def __init__(
            self, images_path, intensity_image_path, spots_image_path, 
            acdc_df_path, segm_path, basename
        ):
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        )
        self.spots_image_path = spots_image_path
    
    def spots_image_data(self):
        return load.load_image_file(self.spots_image_path)
        
class MitoDataSnapshot(_SpotMaxData):
    def __init__(self):
        images_path = os.path.join(
            data_path, 'test_multi_pos_analyse_single_pos', 'Position_15', 
            'Images'
        )
        intensity_image_path = os.path.join(
            images_path, 'ASY15-1_15nM-15_s15_phase_contr.tif'
        )
        spots_image_path = os.path.join(
            images_path, 'ASY15-1_15nM-15_s15_mNeon.tif'
        )
        acdc_df_path = os.path.join(
            images_path, 'ASY15-1_15nM-15_s15_acdc_output.csv'
        )
        segm_path = os.path.join(
            images_path, 'ASY15-1_15nM-15_s15_segm.npz'
        )
        basename = 'ASY15-1_15nM-15_s15_'
        super().__init__(
            images_path, intensity_image_path, spots_image_path, 
            acdc_df_path, segm_path, basename
        )
    