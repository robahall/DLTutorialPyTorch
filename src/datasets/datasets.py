from pathlib import Path
import os
import csv
import functools
from collections import namedtuple
from src import ROOT_DIR

CandidateInfoTuple = namedtuple('CandidateInfoTuple',
                                'isNodule, diameter_mm, series_uid, center_xyz'
                                )


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    data_dir = Path(ROOT_DIR / 'data')
    mhd_list = data_dir.glob('subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[1][:-4] for p in mhd_list}

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
                    isNodule_bool, candidateDiameter_mm , series_uid, candidateCenter_xyz,

                ))
            candidateInfo_list.sort(reverse=True)
            return candidateInfo_list



if __name__ == "__main__":
    getCandidateInfoList()