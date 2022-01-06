"""
(c) MJMJ/2020
"""

import os
import os.path as osp
import copy

def mj_findLatestFileModel(inputdir, pattern, epoch_max=1000):
    '''
    Searchs for check-points during training
    :param inputdir: path
    :param pattern: string compatible with format()
    :return: path to the best file, if any, "" otherwise
    '''

    if epoch_max < 0:
        maxepochs = 1000
    else:
        maxepochs = epoch_max

    bestfile = ""

    for epoch in range(1, maxepochs+1):
        modelname = os.path.join(inputdir, pattern.format(epoch))
        if os.path.isfile(modelname):
            bestfile = copy.deepcopy(modelname)


    return bestfile


def mj_epochOfModelFile(fullpath: str):
    '''
    Parses input string
    :param fullpath: Example: "/tmp/model-state-0101.hdf5"
    :return: string with the version of the model. E.g. "0101"
    '''

    bname = osp.basename(fullpath)
    ln = len(bname.split("-"))
    epoch = bname.split("-")[ln-1].split(".")[0]
    return epoch
