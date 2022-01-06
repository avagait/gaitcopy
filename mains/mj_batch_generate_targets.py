# Tests a gait recognizer CNN
# This version uses a custom DataGenerator

__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'February 2021'

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import os.path as osp
from os.path import expanduser

import pathlib

maindir = pathlib.Path(__file__).parent.absolute()
if sys.version_info[1] >= 6:
    sys.path.insert(0, osp.join(maindir, ".."))
else:
    sys.path.insert(0, str(maindir) + "/..")
homedir = expanduser("~")

sys.path.insert(0, osp.join(maindir,"."))
sys.path.insert(0, osp.join(maindir, ".."))
sys.path.insert(0, osp.join(maindir, "../nets"))
sys.path.insert(0, osp.join(maindir, "../data"))
sys.path.insert(0, osp.join(maindir, "../utils"))

import deepdish as dd
from data.dataGenerator import DataGeneratorGait
from nets.single_model import SingleGaitModel
from nets.mj_gaitcopy_model import GaitCopyModel
from data.mj_augmentation import mj_mirrorsequence
from utils.mj_netUtils import mj_epochOfModelFile

# --------------------------------
import tensorflow as tf

theSEED = 232323
tf.random.set_seed(theSEED)
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # gpu_rate # TODO
tf.executing_eagerly()
graph = tf.Graph()
graph.as_default()

session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()

# --------------------------------

def encodeData(data_generator, model, mirror=False, isOF=True):
    all_vids = []
    all_gt_labs = []
    all_feats = []
    signatures = {}
    nbatches = len(data_generator)
    for bix in range(nbatches):
        data, labels, videoId, cams, fnames = data_generator.__getitemvideoid__(bix)

        if mirror:
            data2 = np.empty(shape=data.shape)
            for tix in range(len(data)):
                data2[tix] = mj_mirrorsequence(data[tix], isOF, True)

            data = np.vstack((data, data2))
            labels = np.vstack((labels, labels))
            videoId = np.hstack((videoId, videoId))
            # Identify the mirror samples
            fnames2 = [fnames[i]+"-mirror" for i in range(len(fnames))]
            fnames = fnames + fnames2

        feats = model.encode(data)

        all_feats.extend(feats)
        all_vids.extend(videoId)
        all_gt_labs.extend(labels[:, 0])

        for i in range(len(feats)):
            signatures[fnames[i]] = feats[i]

    return signatures, all_gt_labs, all_vids


def getSignaturesGaitNet(datadir="matimdbtum_gaid_N150_of25_60x60_lite", nclasses=150, initnet="",
                         modality='of', batchsize=128, use3D=False, camera=0,
                         mean_path="", std_path="", nettype="full", mirror=True, gaitset=False, encode_layer=None):
    # ---------------------------------------
    # Load model
    # ---------------------------------------
    experdir, filename = os.path.split(initnet)

    ep_fix = mj_epochOfModelFile(initnet)

    if nettype == "full":
        model = SingleGaitModel(experdir)
        model.load(initnet, gaitset=gaitset, encode_layer=encode_layer)
    else:
        model = GaitCopyModel(experdir)
        model.load(initnet, compile=False)

    if mean_path != "":
        mean_sample = dd.io.load(mean_path)
    else:
        mean_sample = 0

    if std_path != "":
        std_sample = dd.io.load(std_path)
    else:
        std_sample = 1

    print("* Preparing data...")

    # ---------------------------------------
    # Prepare data
    # ---------------------------------------
    if nclasses == 150:
        data_folder_gallery = osp.join(datadir, 'tfimdb_tum_gaid_N150_train_{}25_60x60'.format(modality))
        info_file_gallery = osp.join(datadir, 'tfimdb_tum_gaid_N150_train_{}25_60x60.h5'.format(modality))
        dataset_info_gallery = dd.io.load(info_file_gallery)
        # Find label mapping for training
        if nclasses > 0:
            ulabels = np.unique(dataset_info_gallery['label'])
            # Create mapping for labels
            labmap_gallery = {}
            for ix, lab in enumerate(ulabels):
                labmap_gallery[int(lab)] = ix
        else:
            labmap_gallery = None
        gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
                                              balanced_classes = False, labmap=labmap_gallery,
                                              modality=modality, datadir=data_folder_gallery, augmentation=False,
                                              use3D=use3D, mirror=False,
                                              mean_sample=mean_sample, std_sample=std_sample, gaitset=gaitset)
    elif nclasses == 155:
        data_folder_gallery = osp.join(datadir, 'tfimdb_tum_gaid_N155_ft_{}25_60x60'.format(modality))
        info_file_gallery = osp.join(datadir, 'tfimdb_tum_gaid_N155_ft_{}25_60x60.h5'.format(modality))
        dataset_info_gallery = dd.io.load(info_file_gallery)
        # Find label mapping for training
        if nclasses > 0:
            ulabels = np.unique(dataset_info_gallery['label'])
            # Create mapping for labels
            labmap_gallery = {}
            for ix, lab in enumerate(ulabels):
                labmap_gallery[int(lab)] = ix
        else:
            labmap_gallery = None
        gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
                                              balanced_classes = False, labmap=labmap_gallery,
                                              modality=modality, datadir=data_folder_gallery, augmentation=False,
                                              use3D=use3D,
                                              mean_sample=mean_sample, std_sample=std_sample, gaitset=gaitset)
    elif nclasses == 16:
        data_folder_gallery = osp.join(datadir, 'tfimdb_tum_gaid_N016_ft_{}25_60x60'.format(modality))
        info_file_gallery = osp.join(datadir, 'tfimdb_tum_gaid_N016_ft_{}25_60x60.h5'.format(modality))
        dataset_info_gallery = dd.io.load(info_file_gallery)
        # Find label mapping for training
        if nclasses > 0:
            ulabels = np.unique(dataset_info_gallery['label'])
            # Create mapping for labels
            labmap_gallery = {}
            for ix, lab in enumerate(ulabels):
                labmap_gallery[int(lab)] = ix
        else:
            labmap_gallery = None
        gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
                                              balanced_classes = False, labmap=labmap_gallery, modality=modality,
                                              datadir=data_folder_gallery, augmentation=False, use3D=use3D,
                                              mirror=False,
                                              mean_sample=mean_sample, std_sample=std_sample, gaitset=gaitset)

        data_folder_n = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_n11-12_{}25_60x60'.format(modality))
        info_file_n = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_n11-12_{}25_60x60.h5'.format(modality))
        dataset_info_n = dd.io.load(info_file_n)
        # Find label mapping for training
        if nclasses > 0:
            ulabels = np.unique(dataset_info_n['label'])
            # Create mapping for labels
            labmap_n = {}
            for ix, lab in enumerate(ulabels):
                labmap_n[int(lab)] = ix
        else:
            labmap_n = None
    elif nclasses == 74:
        data_folder_gallery = osp.join(datadir, 'tfimdb_casia_b_N074_train_{}25_60x60'.format(modality))
        info_file_gallery = osp.join(datadir, 'tfimdb_casia_b_N074_train_{}25_60x60.h5'.format(modality))
        dataset_info_gallery = dd.io.load(info_file_gallery)

        # Find label mapping for training
        if nclasses > 0:
            ulabels = np.unique(dataset_info_gallery['label'])
            # Create mapping for labels
            labmap_gallery = {}
            for ix, lab in enumerate(ulabels):
                labmap_gallery[int(lab)] = ix
        else:
            labmap_gallery = None
        if isinstance(camera, str) and camera == "all":
            cameras_ = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
        else:
            cameras_ = [camera]
        gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
                                              balanced_classes=False,
                                              labmap=labmap_gallery, modality=modality, camera=cameras_,
                                              datadir=data_folder_gallery, augmentation=False, use3D=use3D,
                                              mirror=False,
                                              mean_sample=mean_sample, std_sample=std_sample, gaitset=gaitset)

    elif nclasses == 50:
        data_folder_gallery = osp.join(datadir, 'tfimdb_casia_b_N050_ft_{}25_60x60'.format(modality))
        info_file_gallery = osp.join(datadir, 'tfimdb_casia_b_N050_ft_{}25_60x60.h5'.format(modality))
        dataset_info_gallery = dd.io.load(info_file_gallery)

        # Find label mapping for training
        if nclasses > 0:
            ulabels = np.unique(dataset_info_gallery['label'])
            # Create mapping for labels
            labmap_gallery = {}
            for ix, lab in enumerate(ulabels):
                labmap_gallery[int(lab)] = ix
        else:
            labmap_gallery = None
        cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
        cameras_ = cameras.remove(camera)
        gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
                                              balanced_classes=False,
                                              labmap=labmap_gallery, modality=modality, camera=cameras_,
                                              datadir=data_folder_gallery, augmentation=False, use3D=use3D,
                                              mean_sample=mean_sample, std_sample=std_sample, gaitset=gaitset)
    else:
        sys.exit(0)

    # ---------------------------------------
    # Test data
    # ---------------------------------------
    print("Encoding data...")
    all_feats_gallery, all_gt_labs_gallery, all_vids_gallery = encodeData(gallery_generator, model, mirror=mirror,
                                                                          isOF=modality=='of')

    testdir = os.path.join(experdir, "signatures_"+modality)
    os.makedirs(testdir, exist_ok=True)
    suffix = ""
    if mirror:
        suffix = "_mirror"
    signatures_file = os.path.join(testdir, "signatures_ep{}_{}_{}{}.h5".format(ep_fix, nclasses, camera, suffix))

    the_data = {}
    the_data['signatures'] = all_feats_gallery
    the_data['labels'] = all_gt_labs_gallery
    the_data['videos'] = all_vids_gallery
    the_data['info'] = info_file_gallery
    the_data['labmap'] = labmap_gallery
    dd.io.save(signatures_file, the_data)

    return signatures_file



################# MAIN ################
if __name__ == "__main__":
    import argparse

    # Input arguments
    parser = argparse.ArgumentParser(description='Evaluates a CNN for gait')

    parser.add_argument('--use3d', default=False, action='store_true', help="Use 3D convs in 2nd branch?")

    parser.add_argument('--allcameras', default=False, action='store_true',
                        help="Test with all cameras (only for casia)")

    parser.add_argument('--datadir', type=str, required=False,
                        default=osp.join('/home/GAIT_local/SSD', 'TUM_GAID_tf'),
                        help="Full path to data directory")

    parser.add_argument('--model', type=str, required=True,
                        default=osp.join(homedir,
                                         'experiments/tumgaid_mj_tf/tum150gray_datagen_opAdam_bs128_lr0.001000_dr0.30/model-state-0002.hdf5'),
                        help="Full path to model file (DD: .hdf5)")

    parser.add_argument('--bs', type=int, required=False,
                        default=128,
                        help='Batch size')

    parser.add_argument('--nclasses', type=int, required=True,
                        default=155,
                        help='Maximum number of epochs')

    parser.add_argument('--camera', type=int, required=False,
                        default=90,
                        help='Camera')

    parser.add_argument('--mod', type=str, required=False,
                        default="of",
                        help="gray|depth|of|rgb")

    parser.add_argument('--mean', type=str, required=False,
                        default='',
                        help='Path to mean sample file [.h5]')
    parser.add_argument('--std', type=str, required=False,
                        default='',
                        help='Path to std sample file [.h5]')
    parser.add_argument('--nettype', type=str, required=False,
                        default='full',
                        help='Options: full, copy')

    parser.add_argument('--mirror', default=False, action='store_true', help="Add mirror samples?")

    parser.add_argument('--gaitset', default=False, action='store_true',
                        help="Gaitset model")

    parser.add_argument('--encodelayer', type=str, required=False,
                        default=None,
                        help='Encode layer')

    args = parser.parse_args()
    datadir = args.datadir
    batchsize = args.bs
    nclasses = args.nclasses
    modelpath = args.model
    modality = args.mod
    use3D = args.use3d
    camera = args.camera
    allcameras = args.allcameras
    mean_path = args.mean
    std_path = args.std
    nettype = args.nettype
    mirror = args.mirror
    gaitset = args.gaitset
    encode_layer = args.encodelayer

    # Call the evaluator
    cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
    if allcameras:
        for cam in cameras:
            getSignaturesGaitNet(datadir=datadir, nclasses=nclasses, initnet=modelpath,
                        modality=modality, batchsize=batchsize, use3D=use3D, camera=cam,
                                 mean_path=mean_path, std_path=std_path, mirror=mirror, gaitset=gaitset, encode_layer=encode_layer)
    else:
       signatures_file= getSignaturesGaitNet(datadir=datadir, nclasses=nclasses, initnet=modelpath,
                    modality=modality, batchsize=batchsize, use3D=use3D, camera=camera,
                             mean_path=mean_path, std_path=std_path, nettype=nettype, mirror=mirror, gaitset=gaitset, encode_layer=encode_layer)

    print("Done! "+signatures_file)
