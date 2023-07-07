from . import pipe
from . import printl

class AutoTuneKernel:
    def __init__(self):
        self._image_data = None
        self._segm_data = None
        self._ref_ch_data = None
    
    def set_kwargs(self, kwargs):
        self._kwargs = kwargs
    
    def ref_ch_endname(self):
        return self._kwargs.get('ref_ch_endname', '')
    
    def set_image_data(self, image_data):
        self._image_data = image_data
    
    def set_segm_data(self, segm_data):
        self._segm_data = segm_data
    
    def set_ref_ch_data(self, ref_ch_data):
        self._ref_ch_data = ref_ch_data
    
    def ref_ch_data(self):
        return self._ref_ch_data

    def segm_data(self):
        return self._segm_data
    
    def image_data(self):
        return self._image_data
    
    def _find_best_threshold_method(self):
        result = pipe.spots_semantic_segmentation(
            self._image
        )
    
    def run(self, logger_func=printl):
        pass