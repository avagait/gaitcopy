# Tests a gait recognizer CNN
# This version uses a custom DataGenerator

__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'February 2021'

import os
import sys
import numpy as np

import os.path as osp
from os.path import expanduser

import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

maindir = pathlib.Path(__file__).parent.absolute()
if sys.version_info[1] >= 6:
        sys.path.insert(0, osp.join(maindir, ".."))
else:
        sys.path.insert(0, str(maindir) + "/..")
homedir = expanduser("~")
sys.path.insert(0, homedir + "/gaitmultimodal")
sys.path.insert(0, homedir + "/gaitmultimodal/mains")

import deepdish as dd
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import statistics
from data.dataGenerator import DataGeneratorGait
from nets.mj_gaitcopy_model import GaitCopyModel
from sklearn.neighbors import KNeighborsClassifier
from utils.mj_netUtils import mj_epochOfModelFile

# --------------------------------
import tensorflow as tf

gpu_rate = 0.5
if "GPU_RATE" in os.environ:
    gpu_rate = float(os.environ["GPU_RATE"])

theSEED = 232323
tf.random.set_seed(theSEED)
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = gpu_rate  # TODO
tf.executing_eagerly()
graph = tf.Graph()
graph.as_default()

session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()


# --------------------------------

def encodeData(data_generator, model, modality):
	all_vids = []
	all_gt_labs = []
	all_feats = []
	nbatches = len(data_generator)

	if modality == "of":
		reshape = True
	else:
		reshape = False

	for bix in range(nbatches):
		data, labels, videoId, cams, fname = data_generator.__getitemvideoid__(bix)
		feats = model.encode(data, reshape)

		all_feats.extend(feats)
		all_vids.extend(videoId)
		all_gt_labs.extend(labels[:, 0])

	return all_feats, all_gt_labs, all_vids


def testData(data_generator, model, clf, outpath, outpathres="", save=False):
	all_feats, all_gt_labs, all_vids = encodeData(data_generator, model, modality)

	# Save CM
	if save:
		exper = {}
		exper["feats"] = all_feats
		exper["gtlabs"] = all_gt_labs
		exper["vids"] = all_vids
		dd.io.save(outpath, exper)
		print("Data saved to: " + outpath)

	all_pred_labs = clf.predict(all_feats)
	all_pred_probs = clf.predict_proba(all_feats)

	# Summarize per video
	uvids = np.unique(all_vids)

	# Majority voting per video
	all_gt_labs_per_vid = []
	all_pred_labs_per_vid = []
	all_pred_probs_per_vid = []
	for vix in uvids:
		idx = np.where(all_vids == vix)[0]

		try:
			gt_lab_vid = statistics.mode(list(np.asarray(all_gt_labs)[idx]))
		except:
			gt_lab_vid = np.asarray(all_gt_labs)[idx][0]

		try:
			pred_lab_vid = statistics.mode(list(np.asarray(all_pred_labs)[idx]))
		except:
			pred_lab_vid = np.asarray(all_pred_labs)[idx][0]

		pred_probs_vid = np.mean(np.asarray(all_pred_probs)[idx],axis=0)

		all_gt_labs_per_vid.append(gt_lab_vid)
		all_pred_labs_per_vid.append(pred_lab_vid)
		all_pred_probs_per_vid.append(pred_probs_vid)

	all_gt_labs_per_vid = np.asarray(all_gt_labs_per_vid)
	all_pred_labs_per_vid = np.asarray(all_pred_labs_per_vid)

	# At subsequence level
	M = confusion_matrix(all_gt_labs, all_pred_labs)
	acc = M.diagonal().sum() / len(all_gt_labs)
	print("*** Accuracy [subseq]: {:.2f}".format(acc * 100))

	acc5 = top_k_accuracy_score(all_gt_labs, all_pred_probs, k=5)
	print("*** R5 [subseq]: {:.2f}".format(acc5 * 100))

	# At video level
	Mvid = confusion_matrix(all_gt_labs_per_vid, all_pred_labs_per_vid)
	acc_vid = Mvid.diagonal().sum() / len(all_gt_labs_per_vid)
	print("*** Accuracy [video]: {:.2f}".format(acc_vid * 100))

	acc_vid5 = top_k_accuracy_score(all_gt_labs_per_vid, all_pred_probs_per_vid, k=5)
	print("*** R5 [video]: {:.2f}".format(acc_vid5 * 100))

	# Save results?
	if outpathres != "":
		results = {"accsub": acc, "accvid": acc_vid, "accsub5": acc5, "accvid5": acc_vid5}
		dd.io.save(outpathres, results)


def evalGaitNet(datadir="matimdbtum_gaid_N150_of25_60x60_lite", nclasses=155, initnet="",
                modality='of', batchsize=128, knn=7, use3D=False, camera=0,
				mean_path="", std_path="", gaitset=False):
	# ---------------------------------------
	# Load model
	# ---------------------------------------
	experdir, filename = os.path.split(initnet)

	# Check if results file already exists, so we can skip it
	testdir = os.path.join(experdir, "results")
	os.makedirs(testdir, exist_ok=True)

	# Check if model file exists
	if not osp.exists(initnet):
		print("ERROR: model file does not exist "+initnet)
		return

	epochstr = mj_epochOfModelFile(initnet)
	outpath2_nm = os.path.join(testdir, "results_ep{}_knn_{}_nm_{}_{}.h5".format(epochstr, knn, nclasses, camera))
	if osp.exists(outpath2_nm):
		print("** Results file already exists. Skip it! " + outpath2_nm)
		R = dd.io.load(outpath2_nm)
		print(R["accvid"])
		return

	model = GaitCopyModel(experdir)
	model.load(initnet, compile=False, gaitset=gaitset)

	if mean_path != "":
		mean_sample = dd.io.load(mean_path)
	else:
		mean_sample = 0

	if std_path != "":
		std_sample = dd.io.load(std_path)
	else:
		std_sample = 1

	scenarios = ["N", "B", "S"]
	if nclasses == 50:
		scenarios = ["nm", "bg", "cl"]

	print("* Preparing data...")

	# ---------------------------------------
	# Prepare data
	# ---------------------------------------
	if nclasses == 155:
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
		gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval', labmap=labmap_gallery,
		                                    modality=modality, datadir=data_folder_gallery, augmentation=False, use3D=use3D,
											  mean_sample=mean_sample, std_sample=std_sample)

		data_folder_n = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_n05-06_{}25_60x60'.format(modality))
		info_file_n = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_n05-06_{}25_60x60.h5'.format(modality))
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
		test_generator_n = DataGeneratorGait(dataset_info_n, batch_size=batchsize, mode='test', labmap=labmap_n, modality=modality,
		                                     datadir=data_folder_n, use3D=use3D,
											 mean_sample=mean_sample, std_sample=std_sample)

		data_folder_b = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_b01-02_{}25_60x60'.format(modality))
		info_file_b = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_b01-02_{}25_60x60.h5'.format(modality))
		dataset_info_b = dd.io.load(info_file_b)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_b = {}
			for ix, lab in enumerate(ulabels):
				labmap_b[int(lab)] = ix
		else:
			labmap_b = None
		test_generator_b = DataGeneratorGait(dataset_info_b, batch_size=batchsize, mode='test', labmap=labmap_b, modality=modality,
		                                     datadir=data_folder_b, use3D=use3D,
											 mean_sample=mean_sample, std_sample=std_sample)

		data_folder_s = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_s01-02_{}25_60x60'.format(modality))
		info_file_s = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_s01-02_{}25_60x60.h5'.format(modality))
		dataset_info_s = dd.io.load(info_file_s)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_s = {}
			for ix, lab in enumerate(ulabels):
				labmap_s[int(lab)] = ix
		else:
			labmap_s = None
		test_generator_s = DataGeneratorGait(dataset_info_s, batch_size=batchsize, mode='test', labmap=labmap_s, modality=modality,
		                                     datadir=data_folder_s, use3D=use3D,
											 mean_sample=mean_sample, std_sample=std_sample)
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
		                                      labmap=labmap_gallery, modality=modality,
		                                      datadir=data_folder_gallery, augmentation=False, use3D=use3D,
											  mean_sample=mean_sample, std_sample=std_sample)

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
		test_generator_n = DataGeneratorGait(dataset_info_n, batch_size=batchsize, mode='test', labmap=labmap_n, modality=modality,
		                                     datadir=data_folder_n, use3D=use3D,
											 mean_sample=mean_sample, std_sample=std_sample)

		data_folder_b = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_b03-04_{}25_60x60'.format(modality))
		info_file_b = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_b03-04_{}25_60x60.h5'.format(modality))
		dataset_info_b = dd.io.load(info_file_b)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_b = {}
			for ix, lab in enumerate(ulabels):
				labmap_b[int(lab)] = ix
		else:
			labmap_b = None
		test_generator_b = DataGeneratorGait(dataset_info_b, batch_size=batchsize, mode='test', labmap=labmap_b, modality=modality,
		                                     datadir=data_folder_b, use3D=use3D,
											 mean_sample=mean_sample, std_sample=std_sample)

		data_folder_s = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_s03-04_{}25_60x60'.format(modality))
		info_file_s = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_s03-04_{}25_60x60.h5'.format(modality))
		dataset_info_s = dd.io.load(info_file_s)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_s = {}
			for ix, lab in enumerate(ulabels):
				labmap_s[int(lab)] = ix
		else:
			labmap_s = None
		test_generator_s = DataGeneratorGait(dataset_info_s, batch_size=batchsize, mode='test', labmap=labmap_s, modality=modality,
		                                     datadir=data_folder_s, use3D=use3D,
											 mean_sample=mean_sample, std_sample=std_sample)
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

		if isinstance(camera, str):
			cameras_ = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
			# cameras_ = cameras.remove(camera)
		else:
			cameras_ = [camera]

		gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
		                                      labmap=labmap_gallery, modality=modality, camera=cameras_,
		                                      datadir=data_folder_gallery, augmentation=False, use3D=use3D)

		data_folder_n = osp.join(datadir, 'tfimdb_casia_b_N050_test_nm05-06_{:03d}_{}25_60x60'.format(camera, modality))
		info_file_n = osp.join(datadir, 'tfimdb_casia_b_N050_test_nm05-06_{:03d}_{}25_60x60.h5'.format(camera, modality))
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
		test_generator_n = DataGeneratorGait(dataset_info_n, batch_size=batchsize, mode='test', labmap=labmap_n, modality=modality,
											 camera=cameras_, augmentation=False,
		                                     datadir=data_folder_n, use3D=use3D)

		data_folder_b = osp.join(datadir, 'tfimdb_casia_b_N050_test_bg01-02_{:03d}_{}25_60x60'.format(camera, modality))
		info_file_b = osp.join(datadir, 'tfimdb_casia_b_N050_test_bg01-02_{:03d}_{}25_60x60.h5'.format(camera, modality))
		dataset_info_b = dd.io.load(info_file_b)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_b = {}
			for ix, lab in enumerate(ulabels):
				labmap_b[int(lab)] = ix
		else:
			labmap_b = None
		test_generator_b = DataGeneratorGait(dataset_info_b, batch_size=batchsize, mode='test', labmap=labmap_b, modality=modality,
											 camera=cameras_, augmentation=False,
		                                     datadir=data_folder_b, use3D=use3D)

		data_folder_s = osp.join(datadir, 'tfimdb_casia_b_N050_test_cl01-02_{:03d}_{}25_60x60'.format(camera, modality))
		info_file_s = osp.join(datadir, 'tfimdb_casia_b_N050_test_cl01-02_{:03d}_{}25_60x60.h5'.format(camera, modality))
		dataset_info_s = dd.io.load(info_file_s)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_s = {}
			for ix, lab in enumerate(ulabels):
				labmap_s[int(lab)] = ix
		else:
			labmap_s = None
		test_generator_s = DataGeneratorGait(dataset_info_s, batch_size=batchsize, mode='test', labmap=labmap_s, modality=modality,
											 camera=cameras_, augmentation=False,
		                                     datadir=data_folder_s, use3D=use3D)
	else:
		sys.exit(0)

	# ---------------------------------------
	# Test data
	# ---------------------------------------
	all_feats_gallery, all_gt_labs_gallery, all_vids_gallery = encodeData(gallery_generator, model, modality)
	clf = KNeighborsClassifier(n_neighbors=knn)
	clf.fit(all_feats_gallery, all_gt_labs_gallery)

	print("Evaluating KNN - {}...".format(scenarios[0]))
	testdir = os.path.join(experdir, "results")
	os.makedirs(testdir, exist_ok=True)
	outpath = os.path.join(testdir, "results_knn_{}_nm_{}_{}.h5".format(knn, nclasses, camera))
	testData(test_generator_n, model, clf, outpath, outpath2_nm)

	print("Evaluating KNN - {}...".format(scenarios[1]))
	outpath = os.path.join(testdir, "results_knn_{}_bg_{}_{}.h5".format(knn, nclasses, camera))
	outpath2 = os.path.join(testdir, "results_ep{}_knn_{}_bg_{}_{}.h5".format(epochstr, knn, nclasses, camera))
	testData(test_generator_b, model, clf, outpath, outpath2)

	print("Evaluating KNN - {}...".format(scenarios[2]))
	outpath = os.path.join(testdir, "results_knn_{}_cl_{}_{}.h5".format(knn, nclasses, camera))
	outpath2 = os.path.join(testdir, "results_ep{}_knn_{}_cl_{}_{}.h5".format(epochstr, knn, nclasses, camera))
	testData(test_generator_s, model, clf, outpath, outpath2)


################# MAIN ################
if __name__ == "__main__":
	import argparse

	# Input arguments
	parser = argparse.ArgumentParser(description='Evaluates a CNN for gait')

	parser.add_argument('--use3d', default=False, action='store_true', help="Use 3D convs in 2nd branch?")

	parser.add_argument('--allcameras', default=False, action='store_true', help="Test with all cameras (only for casia)")

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

	parser.add_argument('--knn', type=int, required=True,
	                    default=7,
	                    help='Number of neighbours')

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
	parser.add_argument('--gaitset', default=False, action='store_true',
						help="Gaitset")

	args = parser.parse_args()
	datadir = args.datadir
	batchsize = args.bs
	nclasses = args.nclasses
	modelpath = args.model
	modality = args.mod
	knn = args.knn
	use3D = args.use3d
	camera = args.camera
	allcameras = args.allcameras
	mean_path = args.mean
	std_path = args.std
	gaitset = args.gaitset

	# Call the evaluator
	cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
	if allcameras:
		for cam in cameras:
			evalGaitNet(datadir=datadir, nclasses=nclasses, initnet=modelpath,
			            modality=modality, batchsize=batchsize, knn=knn, use3D=use3D, camera=cam,
						mean_path=mean_path, std_path=std_path, gaitset=gaitset)
	else:
		evalGaitNet(datadir=datadir, nclasses=nclasses, initnet=modelpath,
		            modality=modality, batchsize=batchsize, knn=knn, use3D=use3D, camera=camera,
					mean_path=mean_path, std_path=std_path, gaitset=gaitset)

	print("Done!")

