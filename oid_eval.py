# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import pandas as pd
import pickle
import argparse

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from dataset import dataset_common

'''
VOC2007TEST
    Annotations
    ...
    ImageSets
'''


parser = argparse.ArgumentParser(
                prog='Evaluate OpenImageDatasetV4',
                usage='python oid_eval.py ',
                add_help=True,
                )

parser.add_argument('dataset_path', type=str, help='Path to annotation files.')
parser.add_argument('pred_path', type=str, help='Path to result files.')

args = parser.parse_args()

output_path = os.path.join(args.pred_path, 'eval_output')
anno_files = "validation-annotations-bbox.csv"

def do_python_eval(use_07=True):
    aps = []
    pred_file = 'results_{}.txt' # from 1-num_classes
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    for cls_wid, cls_pair in dataset_common.AYA_LABELS.items():
        if 'none' in cls_wid:
            continue
        cls_id, cls_name = cls_pair
        cls_name = "".join(cls_name.split())
        filename = os.path.join(args.pred_path, pred_file.format(cls_id))
        rec, prec, ap = voc_eval(filename,
                                 os.path.join(args.dataset_path, anno_files),
                                 cls_wid,
                                 ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls_name, ap))
        with open(os.path.join(output_path, cls_name + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             classname,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                               annopath,
                               imagesetfile,
                               classname,
                               [ovthresh],
                               [use_07_metric])
        Top level function that does the PASCAL VOC evaluation.
        detpath: Path to detections
           detpath.format(classname) should produce the detection results file.
        annopath: Path to annotations
           annopath.format(imagename) should be the xml annotations file.
        classname: Category name (duh)
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
           (default False)
    """

    # load annots
    annotations = pd.read_csv(annopath)
    condition = (
                    ((annotations["LabelName"] == "/m/04hgtk") | (annotations["LabelName"] == "/m/0k65p")) & # "Human head" or "Human hand"
                    (annotations["IsOccluded"] == 0) &
                    (annotations["IsGroupOf"] == 0) &
                    (annotations["IsDepiction"] == 0)
                )
    annotations = annotations[condition]
    image_ids = annotations["ImageID"].unique()

    # extract gt objects for this class
    class_recs = {}
    npos = 0

    for image_id in image_ids:
        bbox = annotations[(annotations["ImageID"]==image_id) & (annotations["LabelName"]==classname)][["XMin", "YMin", "XMax", "YMax"]].values

        difficult = np.zeros(len(bbox)).astype(np.bool)
        det = [False] * len(bbox)
        npos = npos + sum(~difficult)
        class_recs[image_id] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    # read dets
    with open(detpath, 'r') as f:
        lines = f.readlines()

    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

if __name__ == '__main__':
        do_python_eval()
