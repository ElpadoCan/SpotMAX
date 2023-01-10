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
from . import config, issues_url, printl, io

class Kernel:
    def __init__(self, debug=False):
        self.logger, self.log_path, self.logs_path = utils.setupLogger('cli')
        self.debug = debug

    @utils.exception_handler_cli
    def init_params(self, params_path, metadata_csv_path=''):
        self._params = config.analysisInputsParams(params_path)
        if metadata_csv_path:
            self._params = io.metadataCSVtoINI(metadata_csv_path, self._params)

    @utils.exception_handler_cli
    def set_metadata(self):
        section = 'METADATA'
        self.PhysicalSizeX = self._params[section]['pixelWidth']
        self.PhysicalSizeY = self._params[section]['pixelHeight']
        self.PhysicalSizeZ = self._params[section]['voxelDepth']
        self.NA = self._params[section]['numAperture']
        self.wavelen = self._params[section]['emWavelens']
        self.z_res_limit = self._params[section]['zResolutionLimit']
        self.yx_multiplier = self._params[section]['yxResolLimitMultiplier']
        self.wavelen = self._params[section]['emWavelen']
        self.SizeT = self._params[section]['SizeT']
        self.SizeZ = self._params[section]['SizeZ']

    @utils.exception_handler_cli
    def preprocess(self, image_data):
        section = 'Pre-processing'
        anchor = 'gaussSigma'
        options = self._params[section][anchor]
        initialVal = options['initialVal']
        sigma = options.get('loadedVal', initialVal)
        self.logger.info(f'Applying a gaussian filter with sigma={sigma}...')
    
    @utils.exception_handler_cli
    def _preproces_ref(self, image_data):
        pass

    @utils.exception_handler_cli
    def segment_ref_ch(self, ref_ch_data=None):
        if ref_ch_data is None:
            ref_ch_path = self._params['File paths and channels']['refChEndName']
            self.check_file_exists(ref_ch_path, desc=' (reference channel)')
            image_data = io.load_image_data()
            image_data = self.preprocess(image_data)


    def check_file_exists(self, file_path, desc=''):
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'The following file does not exist{desc}: "{file_path}"'
            )

    def quit(self, is_error=False):
        print('='*50)
        if is_error:
            err_msg = (
                'spotMAX aborted due to **error**. '
                'More details above or in the folowing log file:\n\n'
                f'{self.log_path}\n\n'
                'You can report this error by opening an issue on our '
                'GitHub page at the following link:\n\n'
                f'{issues_url}\n\n'
                'Please **send the log file** when reporting a bug, thanks!'
            )
            print(err_msg)
        else:
            print(f'spotMAX command line-interface closed.')
        print('='*50)
        exit(utils.get_salute_string())

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
