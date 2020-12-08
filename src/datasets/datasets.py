import csv
import copy
import functools
import timeit
from collections import namedtuple
from pathlib import Path


import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
from torch.utils.data import Dataset

from src import ROOT_DIR
from src.util.util import XyzTuple, xyz2irc
from src.util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

CandidateInfoTuple = namedtuple('CandidateInfoTuple',
                                'isNodule, diameter_mm, series_uid, center_xyz'
                                )


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    data_dir = Path(ROOT_DIR / 'data/luna')
    mhd_list = data_dir.glob('subset*/*.mhd')
    presentOnDisk_set = {p.stem for p in mhd_list}

    diameter_dict = {}
    with open(data_dir / 'annotations.csv', 'r') as file:
        for row in list(csv.reader(file))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )
        candidateInfo_list = []
        with open(data_dir /'candidates.csv', 'r') as file:
            for row in list(csv.reader(file))[1:]:
                series_uid = row[0]

                if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                    continue

                isNodule_bool = bool(int(row[4]))
                candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

                candidateDiameter_mm = 0.0
                for annotation_tup in diameter_dict.get(series_uid, []):
                    annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                    for i in range(3):
                        delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                        if delta_mm > annotationDiameter_mm /4:
                            break
                        else:
                            candidateDiameter_mm = annotationDiameter_mm
                            break

                candidateInfo_list.append(CandidateInfoTuple(
                    isNodule_bool, candidateDiameter_mm, series_uid, candidateCenter_xyz,

                ))
            candidateInfo_list.sort(reverse=True)
            return candidateInfo_list


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

#@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

class Ct:

    def __init__(self, series_uid):
        mhd_path = Path(ROOT_DIR / 'data/luna').glob(f'subset*/{series_uid}.mhd')
        ct_mhd = sitk.ReadImage(str(next(mhd_path)))
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], \
                repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())
        if series_uid:
            self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid == series_uid]
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        log.info(f"{self}: {len(self.candidateInfo_list)} {'validation' if isValSet_bool else 'training'} samples")

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
            not CandidateInfoTuple.isNodule,
            candidateInfo_tup.isNodule

        ], dtype=torch.long
        )

        return candidate_t, pos_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)



def time_luna(n):
    for i in range(n):
        LunaDataset()[i]





if __name__ == "__main__":
    print(LunaDataset()[0])
    #time_luna(1)
    #print(timeit.timeit("time_luna(1)", setup="from __main__ import time_luna"))