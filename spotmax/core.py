import os
import sys
import warnings

import pandas as pd
import numpy as np

import cv2

import skimage.morphology
import skimage.measure
import skimage.transform
import skimage.filters

from . import utils
from . import issues_url, printl, io

class Kernel(io._ParamsParser):
    def __init__(self, debug=False, is_cli=True):
        super().__init__(debug=debug, is_cli=is_cli)
        self.logger, self.log_path, self.logs_path = utils.setupLogger('cli')
        self.debug = debug
        self.is_cli = is_cli
    
    def check_file_exists(self, file_path, desc=''):
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'The following file does not exist{desc}: "{file_path}"'
            )

    @utils.exception_handler_cli
    def preprocess(self, image_data):
        SECTION = 'Pre-processing'
        anchor = 'gaussSigma'
        options = self._params[SECTION][anchor]
        initialVal = options['initialVal']
        sigma = options.get('loadedVal', initialVal)
        self.logger.info(f'Applying a gaussian filter with sigma={sigma}...')
        filtered_data = skimage.filters.gaussian(image_data, sigma=sigma)
    
    @utils.exception_handler_cli
    def _preproces_ref(self, image_data):
        pass
    
    def _load_ref_ch(self):
        ref_ch_path = self._params['File paths and channels']['refChEndName']
        self.check_file_exists(ref_ch_path, desc=' (reference channel)')
        image_data = io.load_image_data()
    
    @utils.exception_handler_cli
    def segment_ref_ch(self, ref_ch_data):
        pass

    @utils.exception_handler_cli
    def load_and_segment_ref_ch(self):
        image_data = self.preprocess(image_data)

    def _run_exp_paths(self, exp_paths):
        for exp_path, exp_info in exp_paths.items():
            exp_path = utils.get_abspath(exp_path)
            run_number = exp_info['run_number']
            pos_foldernames = exp_info['pos_foldernames']  
            spots_ch_endname = exp_info['spotsEndName'] 
            ref_ch_endname = exp_info['refChEndName']
            segm_endname = exp_info['segmEndName']
            ref_ch_segm_endname = exp_info['refChSegmEndName']
            for pos in pos_foldernames:
                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, 'Images')
                import pdb; pdb.set_trace()                

    def _run_single_path(self, single_path_info):
        pass

    @utils.exception_handler_cli
    def run(self, parser_args):
        params_path = parser_args['params']
        metadata_csv_path = parser_args['metadata']
        self.init_params(params_path, metadata_csv_path=metadata_csv_path)
        if self.exp_paths_list:
            for exp_paths in self.exp_paths_list:
                self._run_exp_paths(exp_paths)
        else:
            self._run_single_path(self.single_path_info)
            
    def quit(self, error=None):
        if not self.is_cli and error is not None:
            raise error

        self.logger.info('='*50)
        if error is not None:
            self.logger.info(f'[ERROR]: {error}')
            self.logger.info('^'*50)
            err_msg = (
                'spotMAX aborted due to **error**. '
                'More details above or in the folowing log file:\n\n'
                f'{self.log_path}\n\n'
                'If you cannot solve it, you can report this error by opening '
                'an issue on our '
                'GitHub page at the following link:\n\n'
                f'{issues_url}\n\n'
                'Please **send the log file** when reporting a bug, thanks!'
            )
            self.logger.info(err_msg)
        else:
            self.logger.info(
                'spotMAX command line-interface closed. '
                f'{utils.get_salute_string()}'
            )
            exit()
        self.logger.info('='*50)

def eucl_dist_point_2Dyx(points, all_others):
    """
    Given 2D array of [y, x] coordinates points and all_others return the
    [y, x] coordinates of the two points (one from points and one from all_others)
    that have the absolute minimum distance
    """
    # Compute 3D array where each ith row of each kth page is the element-wise
    # difference between kth row of points and ith row in all_others array.
    # (i.e. diff[k,i] = points[k] - all_others[i])
    diff = points[:, np.newaxis] - all_others
    # Compute 2D array of distances where
    # dist[i, j] = euclidean dist (points[i],all_others[j])
    dist = np.linalg.norm(diff, axis=2)
    # Compute i, j indexes of the absolute minimum distance
    i, j = np.unravel_index(dist.argmin(), dist.shape)
    nearest_point = all_others[j]
    point = points[i]
    min_dist = dist.min()
    return min_dist, nearest_point

def rotationalVolume(obj, PhysicalSizeY=1, PhysicalSizeX=1):
    vox_to_fl = float(PhysicalSizeY)*(float(PhysicalSizeX)**2)
    rotate_ID_img = skimage.transform.rotate(
        obj.image.astype(np.uint8), -(obj.orientation*180/np.pi),
        resize=True, order=3, preserve_range=True
    )
    radii = np.sum(rotate_ID_img, axis=1)/2
    vol_vox = np.sum(np.pi*(radii**2))
    if vox_to_fl is not None:
        return vol_vox, float(vol_vox*vox_to_fl)
    else:
        return vol_vox, vol_vox

def calcMinSpotSize(
        emWavelen, NA, physicalSizeX, physicalSizeY, physicalSizeZ,
        zResolutionLimit_um, yxResolMultiplier
    ):
    try:
        airyRadius_nm = (1.22 * emWavelen)/(2*NA)
        airyRadius_um = airyRadius_nm*1E-3
        yxMinSize_um = airyRadius_um*yxResolMultiplier
        xMinSize_pxl = yxMinSize_um/physicalSizeX
        yMinSize_pxl = yxMinSize_um/physicalSizeY
        zMinSize_pxl = zResolutionLimit_um/physicalSizeZ
        zyxMinSize_pxl = (zMinSize_pxl, yMinSize_pxl, xMinSize_pxl)
        zyxMinSize_um = (zResolutionLimit_um, yxMinSize_um, yxMinSize_um)
        return zyxMinSize_pxl, zyxMinSize_um
    except ZeroDivisionError as e:
        # warnings.warn(str(e), RuntimeWarning)
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

def skeletonize(dataToSkel, is_zstack=False):
    skeleton = skimage.morphology.skeletonize(dataToSkel)
    skeletonCoords = {'all': np.nonzero(skeleton)}
    if is_zstack:
        for z, skel in enumerate(skeleton):
            skeletonCoords[z] = np.nonzero(skel)
    return skeletonCoords

def objContours(obj):
    contours, _ = cv2.findContours(
        obj.image.astype(np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )
    min_y, min_x, _, _ = obj.bbox
    contours_li = []
    for cont in contours:
        cont = np.squeeze(cont, axis=1)
        cont = np.vstack((cont, cont[0]))
        cont += [min_x, min_y]
        contours_li.append(cont)
    return contours_li

def findContours(dataToCont, is_zstack=False):
    contCoords = {'proj': {}}
    if is_zstack:
        for z, img in enumerate(dataToCont):
            lab = skimage.measure.label(img)
            rp = skimage.measure.regionprops(lab)
            allObjContours = {}
            for obj in rp:
                contours_li = objContours(obj)
                allObjContours[obj.label] = contours_li
            contCoords[z] = allObjContours
        dataToCont2D = dataToCont.max(axis=0)
    else:
        dataToCont2D = dataToCont.max(axis=0)

    lab = skimage.measure.label(dataToCont2D)
    rp = skimage.measure.regionprops(lab)
    for obj in rp:
        contours_li = objContours(obj)
        contCoords['proj'][obj.label] = contours_li
    return contCoords
