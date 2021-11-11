import os
import sys

import pandas as pd
import numpy as np

import cv2

import skimage.morphology
import skimage.measure

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
        cv2.CHAIN_APPROX_SIMPLE
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
        dataToCont2D = dataToSkel.max(axis=0)

    lab = skimage.measure.label(dataToCont2D)
    rp = skimage.measure.regionprops(lab)
    for obj in rp:
        contours_li = objContours(obj)
        contCoords['proj'][obj.label] = contours_li
    return contCoords
