import numpy as np
import os
import pandas as pd
import shutil
import math
from diagnostics.utils import load_marker_csv
from scipy.interpolate import interp2d

import numpy as np

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def do_confidence_threshold(coordinate, confidence, confidence_threshold=.95):
    low_confidence = np.where(confidence < confidence_threshold)[0]
    coordinate[low_confidence] = math.nan
    nans, x = nan_helper(coordinate)
    coordinate[nans]= np.interp(x(nans), x(~nans), coordinate[~nans])
    return coordinate, low_confidence
    
def do_ensemble_confidence_threshold(coordinate, confidence, confidence_thresholds=[.8, .85, .9, .95]):
    coordinates_list = []
    low_confidences = []
    for confidence_threshold in confidence_thresholds:
        coordinate_copy = coordinate.copy()
        low_confidence = np.where(confidence < confidence_threshold)[0]
        coordinate_copy[low_confidence] = math.nan
        nans, x = nan_helper(coordinate_copy)
        coordinate_copy[nans]= np.interp(x(nans), x(~nans), coordinate_copy[~nans])
        coordinates_list.append(coordinate_copy)
        low_confidences.append(low_confidence)
    coordinates_array = np.asarray(coordinates_list)
    coordinate = np.mean(coordinates_array, axis=0)
    return coordinate, low_confidences

def postprocess_traces(csv, method='confidence', excluded_keypoints=[], verbose=True, **kwargs):
    x, y, confidence, marker_names = load_marker_csv(csv)
    x_postprocess = []
    y_postprocess = []
    for marker_id in range(x.shape[1]):
        for coordinate_name, coordinate in (("x", x), ("y", y)):
            coordinate_curr = coordinate.T[marker_id].copy()
            confidence_curr = confidence.T[marker_id].copy()
            if marker_names[marker_id] in excluded_keypoints:
                if verbose:
                    print("skipping " + marker_names[marker_id] + "_" + coordinate_name)
            #confidence thresholding
            elif method=='confidence':
                if verbose:
                    print("confidence thresholding " + marker_names[marker_id] + "_" + coordinate_name)
                assert "confidence_threshold" in kwargs.keys(), "confidence_threshold not in kwargs"
                confidence_threshold = kwargs["confidence_threshold"]
                coordinate_curr, low_confidence_curr = do_confidence_threshold(coordinate_curr, confidence_curr, confidence_threshold)    
            #ensemble confidence thresholding
            elif method=='confidence_ensemble':
                if verbose:
                    print("ensemble confidence thresholding " + marker_names[marker_id] + "_" + coordinate_name)
                assert "confidence_thresholds" in kwargs.keys(), "confidence_thresholds not in kwargs"
                confidence_thresholds = kwargs["confidence_thresholds"]
                coordinate_curr, low_confidences_curr = do_ensemble_confidence_threshold(coordinate_curr, confidence_curr, confidence_thresholds)   
            #store result
            else:
                raise NotImplementedError(str(method) + " is not a supported postprocessing method")
            if coordinate_name == "x":
                x_postprocess.append(coordinate_curr)
            else:
                y_postprocess.append(coordinate_curr)
    x_postprocess = np.asarray(x_postprocess).T
    y_postprocess = np.asarray(y_postprocess).T
    coordinates_postprocess = np.stack((x_postprocess, y_postprocess), axis=2)
    return coordinates_postprocess, confidence
