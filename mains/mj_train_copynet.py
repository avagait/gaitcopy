# Trains a gait recognizer CNN
# This version uses a custom DataGenerator

__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'February 2021'

import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

import os.path as osp
from os.path import expanduser

import pathlib

maindir = pathlib.Path(__file__).parent.absolute()

homedir = expanduser("~")

# print(maindir)
sys.path.insert(0, osp.join(maindir,"."))
sys.path.insert(0, osp.join(maindir, ".."))
sys.path.insert(0, osp.join(maindir, "../nets"))

# --------------------------------
gpu_rate = 0.9
if "GPU_RATE" in os.environ:
    gpu_rate = float(os.environ["GPU_RATE"])

theSEED = 232323
tf.random.set_seed(theSEED)
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = gpu_rate  # gpu_rate # TODO

tf.executing_eagerly()
graph = tf.Graph()
graph.as_default()

session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()
# --------------------------------

from tensorflow.keras import optimizers

import deepdish as dd
from nets.mj_gaitcopy_model import GaitCopyModel
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from data.mj_dataGeneratorCopyNet import DataGeneratorCopyNet
from utils.mj_netUtils import mj_findLatestFileModel
from callbacks.model_saver import ModelSaver
from tensorboard.plugins import projector


# ===============================================================
def trainGaitCopyNet(datadir="matimdbtum_gaid_N150_of25_60x60_lite", experfix="of", targets_file="INVALID", nclasses=0,
					lr=0.01, dropout=0.4, experdirbase=".", epochs=15, batchsize=150, optimizer="SGD",
					loss_mode = "MSE", wtri = 0.1, modality="of", initnet="", use3D=False, freeze_all=False,
					nofreeze=False, logdir="", extra_epochs=0, model_version='iwann', cameras=None, sufix=None,
					mean_path = "", std_path = "", with_l2 = False, kinit='glorot_uniform', margintri=0.25,
					lrmode="vdec", mirror=True, augmentation=False, noise=0, valprop=0.06, hdelta=0.5,
					cross_signatures=False, drop_code = 0, alphamob=1, gpool='avg', xsiglevel=0, verbose=0, pxk=False, k_size=4):
	"""
	Trains a CNN for gait recognition
	:param datadir: root dir containing dd files
	:param experfix: string to customize experiment name
	:param nclasses: number of classes
	:param lr: starting learning rate
	:param dropout: dropout value
	:param tdim: extra dimension of samples. Usually 50 for OF, and 25 for gray and depth
	:param epochs: max number of epochs
	:param batchsize: integer
	:param optimizer: options like {SGD, Adam,...}
	:param logdir: path to save Tensorboard info
	:param ndense_units: number of dense units for last FC layer
	:param mean_path: path to mean sample file (h5)
	:param verbose: integer
	:return: model, experdir, accuracy
	"""

	if use3D:
		if modality == 'of':
			input_shape = (25, 60, 60, 2)
		else:		# Gray
			input_shape = (25, 60, 60, 1)
	else:
		if modality == 'of':
			input_shape = (50, 60, 60)
		else:		# Gray
			input_shape = (25, 60, 60)

	mobpars = {'alpha': alphamob, 'gpool': gpool}  # For mobilenet
	if model_version == 'iwann':
		number_convolutional_layers = 4
		filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
		filters_numbers = [96, 192, 512, 4096]
		ndense_units = 2048
		strides = [1, 2, 1, 1]
	elif model_version == 'bmvc':
		if use3D:
			filters_size = [(3, 5, 5), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 2, 2), (2, 1, 1)]
			filters_numbers = [64, 128, 256, 512, 512, 512]
		else:
			filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
			filters_numbers = [96, 192, 512, 512]
		number_convolutional_layers = len(filters_numbers)
		if nclasses == 74 or nclasses == 50:
			ndense_units = 2048
		else:
			ndense_units = 1024
		strides = [1, 1, 1, 1]
	elif model_version == 'smallA':		
		if use3D:
			filters_size = [(3, 5, 5), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 2, 2), (2, 1, 1)]
			filters_numbers = [64, 96, 128, 256, 256, 512]
		else:
			filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
			filters_numbers = [64, 128, 256, 512]
		number_convolutional_layers = len(filters_numbers)
		if nclasses == 74 or nclasses == 50:
			ndense_units = 2048
		else:
			ndense_units = 1024
		strides = [1, 1, 1, 1]
	elif model_version == 'smallB':		
		if use3D:
			filters_size = [(3, 5, 5), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 2, 2), (2, 1, 1)]
			filters_numbers = [48, 96, 128, 128, 256, 512]
		else:
			filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
			filters_numbers = [64, 128, 256, 512]
		number_convolutional_layers = len(filters_numbers)
		if nclasses == 74 or nclasses == 50:
			ndense_units = 2048
		else:
			ndense_units = 1024
		strides = [1, 1, 1, 1]		
	elif model_version == 'bmvcfc':
		number_convolutional_layers = 4
		filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
		filters_numbers = [96, 160, 512, 512]
		if nclasses == 74 or nclasses == 50:
			ndense_units = 2048
		else:
			ndense_units = 1024
		strides = [1, 1, 1, 1]
	elif model_version == 'mobilenet3d': 		
		if nclasses == 74 or nclasses == 50:
			ndense_units = 2048
		else:
			ndense_units = 1024

		# Fake vars
		number_convolutional_layers = 1
		filters_size = [(1,1,1)]
		filters_numbers = [32]		
		strides = [1]
	else:
		number_convolutional_layers = 4
		filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
		filters_numbers = [96, 192, 512, 4096]
		ndense_units = 2048
		strides = [1, 2, 1, 1]

	weight_decay = 0.00005
	momentum = 0.9
	if 'mobile' in model_version:
		weight_decay = 1e-03

	optimfun = optimizers.Adam(lr=lr)
	infix = "_opAdam"
	if optimizer != "Adam":
		infix = "_op" + optimizer
		if optimizer == "SGD":
			optimfun = optimizers.SGD(lr=lr, momentum=momentum)
		elif optimizer == "AMSGrad":
			optimfun = optimizers.Adam(lr=lr, amsgrad=True)
		else:
			optimfun = eval("optimizers." + optimizer + "(lr=lr)")

	initialLR = lr
	lrsteps = np.concatenate((initialLR*np.ones(50), initialLR/10*np.ones(40), initialLR/100*np.ones(30)))
	if epochs > 100:
		lrsteps = np.concatenate((lrsteps, initialLR / 1000 * np.ones(epochs - 120)))

	if lrmode == "cte":
		reduce_lr = LearningRateScheduler(lambda x: lrsteps[x])
	else:  # lrmode == "vdec":
		reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.2,
	                              patience=3, min_lr=0.00001, verbose=2)
	
	es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=75, verbose=1)

	if loss_mode != "MSE":
			infix = infix + "_"+loss_mode
	if loss_mode == "sL1tri" or loss_mode == "sL1triH" or loss_mode == "sL1triB":
		multiloss = True
		loss_weights = [1.0, wtri]

		if wtri != 0.1:
			infix = infix + "{:03d}".format(int(wtri*100))

		if margintri != 0.25:
			infix = infix + "m{:03d}".format(int(margintri * 100))

		if with_l2:
			infix = infix + "L2"
	else:
		multiloss = False
		loss_weights = [1.0]
		if with_l2:
			print("WARN: L2-norm has been disabled as loss does not include TripletLoss")
			with_l2 = False

	if ('sL1' in loss_mode or loss_mode == 'Huber') and hdelta != 0.5:
		infix = infix + "d{:02d}".format(int(hdelta*10))
	


	if use3D:
		f3d = ""
		if kinit != 'glorot_uniform':
			f3d = kinit[0:2]
		infix = "_" + modality + "3D"+f3d + infix

	else:
		infix = "_" + modality + infix

	if nofreeze:
		freeze_convs = False
	else:
		freeze_convs = True
	if initnet != "" and freeze_all:
		infix = infix + "_frall"

	if mean_path != "":
		mean_sample = dd.io.load(mean_path)
		infix = infix + "_mean"
	else:
		mean_sample = 0

	if std_path != "":
		std_sample = dd.io.load(std_path)
		infix = infix + "_std"
	else:
		std_sample = 1

	if mirror:
		infix = infix + "_mir"

	if augmentation:
		infix = infix + "_aug"

	if noise > 0:
		infix = infix + "_noi{:.2f}".format(noise)

	if cross_signatures or xsiglevel != 0:
		infix = infix + "_xs"
		if xsiglevel != 0:
			infix = infix + "{}".format(xsiglevel)

	if pxk:
		infix = infix + "_pxk"

	if valprop > 0:
		infix = infix + "_val{:02d}".format(int(valprop*100))

    # "cte"
	lrmodefix = ""
	if lrmode != "vdec":
		lrmodefix = "cte"

	dropoutfix = "{:0.2f}".format(dropout)
	if drop_code > 0:
		dropoutfix = dropoutfix + "+{:0.2f}".format(drop_code)

	modelvfix = model_version
	if "mobile" in model_version:
		if alphamob != 1:
			modelvfix = modelvfix + 'a{:0.1f}'.format(alphamob)
		if gpool != 'avg':
			modelvfix = modelvfix + gpool


	# Create a TensorBoard instance with the path to the logs directory
	subdir = experfix + '_' + modelvfix + '_N{:03d}_datagen{}_bs{:03d}_lr{:0.6f}{}_dr{}'.format(nclasses, infix, batchsize, lr,
	                                                                             lrmodefix, dropoutfix)  # To be customized
	if sufix is not None:
		subdir = subdir + "_" + sufix
	experdir = osp.join(experdirbase, subdir)
	if verbose > 0:
		print(experdir)
	if not osp.exists(experdir):
		os.makedirs(experdir)

	# Prepare model
	initepoch = 0
	pattern_file = "model-state-{:04d}.hdf5"
	previous_model = mj_findLatestFileModel(experdir, pattern_file, epoch_max=epochs)
	print(previous_model)
	if os.path.exists(os.path.join(experdir, 'model-final.hdf5')):
		print("Already trained model, skipping.")
		return None, None
	else:
		model = GaitCopyModel(experdir)
		#model = GaitCopyModel(experdir)
		if previous_model != "":
			pms = previous_model.split("-")
			initepoch = int(pms[len(pms) - 1].split(".")[0])
			print("* Info: a previous model was found. Warming up from it...[{:d}]".format(initepoch))
			from tensorflow.keras.models import load_model
			model.load(previous_model)
		else:
			if initnet != "":
				print("* Model will be init from: " + initnet)

			model.build_or_load(input_shape, number_convolutional_layers, filters_size, filters_numbers, strides,
			                    ndense_units, weight_decay, dropout, optimizer=optimfun, nclasses=nclasses,
			                    initnet=initnet, freeze_convs=freeze_convs, use3D=use3D, freeze_all=freeze_all,
								model_version=model_version, loss_mode=loss_mode, loss_weights=loss_weights,
								margin=margintri, drop_code=drop_code,
								with_l2=with_l2, kinit=kinit, mobpars=mobpars)

		model.model.summary()

		# Tensorboard
		if logdir == "":
			logdir = experdir
			model_saver = ModelSaver(model, every=5)

			from tensorflow.keras.callbacks import TensorBoard

			tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False,
			                          profile_batch=0)
			callbacks = [reduce_lr, tensorboard, model_saver, es_callback]
		else:  # This case is for parameter tuning
			# Save checkpoint
			model_saver = ModelSaver(model, every=5)

			from tensorflow.keras.callbacks import TensorBoard
			tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False,
			                          profile_batch=3)

			callbacks = [reduce_lr, tensorboard, model_saver]

		# ---------------------------------------
		# Prepare data
		# ---------------------------------------
		if nclasses == 150:
			data_folder = osp.join(datadir, 'tfimdb_tum_gaid_N150_train_{}25_60x60'.format(modality))
			info_file = osp.join(datadir, 'tfimdb_tum_gaid_N150_train_{}25_60x60.h5'.format(modality))
		elif nclasses == 155:
			data_folder = osp.join(datadir, 'tfimdb_tum_gaid_N155_ft_{}25_60x60'.format(modality))
			info_file = osp.join(datadir, 'tfimdb_tum_gaid_N155_ft_{}25_60x60.h5'.format(modality))
		elif nclasses == 16:
			data_folder = osp.join(datadir, 'tfimdb_tum_gaid_N016_ft_{}25_60x60'.format(modality))
			info_file = osp.join(datadir, 'tfimdb_tum_gaid_N016_ft_{}25_60x60.h5'.format(modality))
		elif nclasses == 74:
			data_folder = osp.join(datadir, 'tfimdb_casia_b_N074_train_{}25_60x60'.format(modality))
			info_file = osp.join(datadir, 'tfimdb_casia_b_N074_train_{}25_60x60.h5'.format(modality))
		elif nclasses == 50:
			data_folder = osp.join(datadir, 'tfimdb_casia_b_N050_ft_{}25_60x60'.format(modality))
			info_file = osp.join(datadir, 'tfimdb_casia_b_N050_ft_{}25_60x60.h5'.format(modality))
		else:
			sys.exit(0)

		dataset_info = dd.io.load(info_file)

		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info['label'])
			# Create mapping for labels
			labmap = {}
			for ix, lab in enumerate(ulabels):
				labmap[int(lab)] = ix
		else:
			labmap = None
			ulabels = [0]
		
		nids = len(ulabels)
		nids_val = int(0.06*nids)
		print("Using {} subjects for validation".format(nids_val))
		ids_val = ulabels[0:nids_val]
		ids_train = ulabels[nids_val:nids]

		for t_id in ids_val:
			the_idxs = np.where(dataset_info['label'] == t_id)[0]
			dataset_info['set'][the_idxs] = 2

		print(sum(dataset_info['set'] == 2))

		for t_id in ids_train:
			the_idxs = np.where(dataset_info['label'] == t_id)[0]
			dataset_info['set'][the_idxs] = 1

		# Load target data: to be copied by the network
		targets_dict = dd.io.load(targets_file)

		# Data generators
		train_generator = DataGeneratorCopyNet(dataset_info, batch_size=batchsize, mode='train', labmap=labmap,
											   targets=targets_dict['signatures'], use3D=use3D,
											modality=modality, datadir=data_folder, camera=cameras,
											mean_sample=mean_sample, std_sample=std_sample,
											   mirror = mirror, noise=noise,
											   augmentation=augmentation, multiloss=multiloss,
											   nids=nclasses, cross_signatures=cross_signatures, pxk=pxk, k=k_size)

		train_generator.on_epoch_end()
		
		val_generator = DataGeneratorCopyNet(dataset_info, batch_size=batchsize, mode='val', labmap=labmap,
											 targets=targets_dict['signatures'], use3D=use3D,
										  modality=modality, datadir=data_folder, camera=cameras,
										  mean_sample=mean_sample, std_sample=std_sample,
											 mirror=mirror, noise=noise,
											 augmentation=augmentation, multiloss=multiloss,
											 nids=nclasses, cross_signatures=False, pxk=pxk, k=k_size)  # mjmarin: don't want to mix signatures during val


		# ---------------------------------------
		# Train model
		# --------------------------------------
		if verbose > 1:
			print(experdir)

		last_epoch = model.fit(epochs, callbacks, train_generator, val_generator, current_step=initepoch)

		# Fine-tune on remaining validation samples
		if extra_epochs > 0:
			if verbose > 0:
				print("Adding validation samples to training and run for few epochs...")
			del train_generator

			train_generator = DataGeneratorCopyNet(dataset_info, batch_size=batchsize, mode='trainval', labmap=labmap, modality=modality,
												   targets=targets_dict['signatures'], use3D=use3D,
			                                    datadir=data_folder, camera=cameras,
												mean_sample=mean_sample, std_sample=std_sample,
												   mirror = mirror, noise=noise,
												   augmentation=augmentation, multiloss=multiloss,
												   nids=nclasses, pxk=pxk, k=k_size)

			ft_epochs = last_epoch + extra_epochs  # DEVELOP!
			the_min_lr = 0.00001
			callbacks[0] = ReduceLROnPlateau(monitor='mean_absolute_error', factor=0.2,
			                                 patience=3, min_lr=the_min_lr)
			tf.keras.backend.set_value(model.model.optimizer.lr, max(the_min_lr, model.model.optimizer.lr*0.1))
			model.fit(ft_epochs, callbacks, train_generator, val_generator, initepoch+last_epoch)

		# Save codes to Projector
		print("Exporting to projector")
		META_DATA_FNAME = 'meta.tsv'  # Labels will be stored here
		EMBEDDINGS_TENSOR_NAME = 'embeddings'
		os.makedirs(logdir, exist_ok=True)
		EMBEDDINGS_FPATH = os.path.join(logdir, EMBEDDINGS_TENSOR_NAME + '.ckpt')
		STEP = epochs + extra_epochs

		data = []
		labels = []
		for e in range(0, len(train_generator)):
			_X, _Y = train_generator.__getitem__(e)
			_X_codes = model.encode(_X)
			data.extend(_X_codes)
			labels.extend(_Y)

		mj_register_embedding(EMBEDDINGS_TENSOR_NAME, META_DATA_FNAME, logdir)
		mj_save_labels_tsv(labels, META_DATA_FNAME, logdir)

		tensor_embeddings = tf.Variable(data, name=EMBEDDINGS_TENSOR_NAME)
		saver = tf.compat.v1.train.Saver([tensor_embeddings])  # Must pass list or dict
		saver.save(sess=None, global_step=STEP, save_path=EMBEDDINGS_FPATH)

		return model, experdir


def mj_register_embedding(embedding_tensor_name, meta_data_fname, log_dir, sprite_path=None, image_size=(1, 1)):
	config = projector.ProjectorConfig()
	embedding = config.embeddings.add()
	embedding.tensor_name = embedding_tensor_name
	embedding.metadata_path = meta_data_fname

	if sprite_path:
		embedding.sprite.image_path = sprite_path
		embedding.sprite.single_image_dim.extend(image_size)

	projector.visualize_embeddings(log_dir, config)


def mj_save_labels_tsv(labels, filepath, log_dir):
	with open(os.path.join(log_dir, filepath), 'w') as f:
		# f.write('Class\n') # Not allowed as we have just one column
		for label in labels:
			f.write('{}\n'.format(label))


################# MAIN ################
if __name__ == "__main__":
	import argparse

	# Input arguments
	parser = argparse.ArgumentParser(description='Trains a CNN for gait')
	parser.add_argument('--debug', default=False, action='store_true')
	parser.add_argument('--use3d', default=False, action='store_true', help="Use 3D convs in 2nd branch?")
	parser.add_argument('--freezeall', default=False, action='store_true', help="Freeze all weights?")
	parser.add_argument('--nofreeze', default=False, action='store_true', help="Avoid freezing any weight?")
	parser.add_argument('--l2', default=False, action='store_true', help="Apply L2 norm signature for triplet loss?")
	parser.add_argument('--dropout', type=float, required=False,
	                    default=0.4, help='Dropout value for after-fusion layers')
	parser.add_argument('--dropcode', type=float, required=False,
	                    default=0.0, help='Dropout right after the code, before L2 norm')
	parser.add_argument('--lr', type=float, required=False,
	                    default=0.01,
	                    help='Starting learning rate')
	parser.add_argument('--datadir', type=str, required=False,
	                    default=osp.join('/home/GAIT_local/SSD', 'TUM_GAID_tf'),
	                    help="Full path to data directory")
	parser.add_argument('--experdir', type=str, required=True,
	                    default=osp.join(homedir, 'experiments', 'tumgaid_gaitcopy'),
	                    help="Base path to save results of training")
	parser.add_argument('--targets', type=str, required=True,
	                    default=osp.join(homedir, 'experiments', 'tumgaid_gaitcopy'),
	                    help="Path to file containing the target signatures to be copied")
	parser.add_argument('--prefix', type=str, required=True,
	                    default="demo",
	                    help="String to prefix experiment directory name.")
	parser.add_argument('--bs', type=int, required=False,
	                    default=150,
	                    help='Batch size')
	parser.add_argument('--epochs', type=int, required=False,
	                    default=250,
	                    help='Maximum number of epochs')
	parser.add_argument('--extraepochs', type=int, required=False,
	                    default=50,
	                    help='Extra number of epochs to add validation data')
	parser.add_argument('--nclasses', type=int, required=True,
	                    default=150,
	                    help='Maximum number of epochs')
	parser.add_argument('--camera', type=int, required=False,
						default=90,
						help='Single camera used')
	parser.add_argument('--model_version', type=str, required=False,
	                    default='bmvc',
	                    help='Model version. [iwann, bmvc]')
	parser.add_argument('--tdim', type=int, required=False,
	                    default=50,
	                    help='Number of dimensions in 3rd axis time. E.g. OF=50')
	parser.add_argument('--optimizer', type=str, required=False,
	                    default="Adam",
	                    help="Optimizer: SGD, Adam, AMSGrad")
	parser.add_argument('--loss', type=str, required=False,
	                    default="sL1",
	                    help="Loss function: MSE, sL1, sL1tri, sL1triH")
	parser.add_argument('--kinit', type=str, required=False,
	                    default='glorot_uniform',
	                    help="3DConv kernel initializer: glorot_uniform, he_uniform")			
			
	parser.add_argument('--wtri', type=float, required=False,
						default=0.1,
						help='Weight of TripletLoss, if multiloss')
	parser.add_argument('--margintri', type=float, required=False,
						default=0.25,
						help='Margin of TripletLoss, if multiloss')
	parser.add_argument('--hdelta', type=float, required=False,
						default=0.5,
						help='Huber delta')
	parser.add_argument('--mod', type=str, required=False,
	                    default="of",
	                    help="Input modality: of, gray, depth")
	parser.add_argument('--initnet', type=str, required=False,
	                    default="",
	                    help="Path to net to initialize")
	parser.add_argument('--mean', type=str, required=False,
						default='',
						help='Path to mean sample file [.h5]')
	parser.add_argument('--std', type=str, required=False,
						default='',
						help='Path to std sample file [.h5]')
	parser.add_argument('--mirror', default=False, action='store_true', help="Add mirror samples?")
	parser.add_argument('--aug', default=False, action='store_true', help="Use data augmentation?")
	parser.add_argument('--noise', type=float, required=False,
						default=0,
						help='Level of random noise for target signatures. Def. 0=none. E.g. 0.1')
	parser.add_argument('--xs', default=False, action='store_true', help="Use cross-signatures?")
	parser.add_argument("--xsl", type=int,
	                    default=0, help="Cross-signatures level")
	parser.add_argument("--verbose", type=int,
	                    nargs='?', const=False, default=1,
	                    help="Whether to enable verbosity of output")
	parser.add_argument('--lrmode', type=str, required=False,
						default='vdec',
						help='Choose: vdec, cte')
	parser.add_argument('--pool', type=str, required=False,
						default='avg',
						help='For mobilenets. Choose: avg, max')
	parser.add_argument('--alpha', type=float, required=False,
						default='1',
						help='For mobilenets, expand factor. Def. 1.0')
	parser.add_argument('--pxk', default=False, action='store_true', help="PxK?")
	parser.add_argument('--k_size', type=int, required=False,
						default=16,
						help='K size')


	args = parser.parse_args()
	verbose = args.verbose
	dropout = args.dropout
	dropcode = args.dropcode
	datadir = args.datadir
	prefix = args.prefix
	epochs = args.epochs
	extraepochs = args.extraepochs
	batchsize = args.bs
	nclasses = args.nclasses
	model_version = args.model_version
	lr = args.lr
	tdim = args.tdim
	optimizer = args.optimizer
	loss_mode = args.loss
	experdirbase = args.experdir
	modality = args.mod
	use3D = args.use3d
	IS_DEBUG = args.debug
	freeze_all = args.freezeall
	nofreeze = args.nofreeze
	mean_path = args.mean
	std_path = args.std
	targets_file = args.targets
	initnet = args.initnet
	wtri = args.wtri
	margintri = args.margintri
	hdelta = args.hdelta
	with_l2 = args.l2
	kinit = args.kinit
	camera = args.camera
	mirror = args.mirror
	augmentation = args.aug
	noise = args.noise
	xs = args.xs
	xsl = args.xsl
	lrmode= args.lrmode
	gpool = args.pool
	alphamob = args.alpha
	pxk = args.pxk
	k_size = args.k_size

	# Start the processing
	if nclasses == 50:
		# Train as many models as cameras.
		cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
		for camera in cameras:
			cameras_ = cameras.copy()
			cameras_.remove(camera)
			print("Fine tuning with ", cameras_, " cameras")
			final_model, experdir = trainGaitCopyNet(datadir=datadir, experfix=prefix, lr=lr, dropout=dropout,
			                                     experdirbase=experdirbase, nclasses=nclasses, optimizer=optimizer,
													 loss_mode=loss_mode, wtri=wtri,
			                                     epochs=epochs, batchsize=batchsize, logdir="", modality=modality,
			                                     initnet=initnet, use3D=use3D, freeze_all=freeze_all, nofreeze=nofreeze,
			                                     extra_epochs=extraepochs, model_version=model_version,
			                                     cameras=cameras_, mirror= mirror, augmentation=augmentation,
			                                      lrmode=lrmode, drop_code = dropcode,
													 noise=noise, margintri=margintri, cross_signatures=xs,
			                                     sufix=str(camera).zfill(3), verbose=verbose, pxk=pxk, k_size=k_size)
			if final_model is not None:
				final_model.save()
	else:
		if nclasses == 74:
			cameras = [camera]   # DEVELOP!
			suffix = "c{:03d}".format(camera)
		else:
			cameras = None
			suffix = None
		final_model, experdir = trainGaitCopyNet(datadir=datadir, experfix=prefix, targets_file=targets_file,
												 lr=lr, dropout=dropout,
		                                     experdirbase=experdirbase, nclasses=nclasses, optimizer=optimizer,
												 loss_mode=loss_mode, wtri=wtri,
		                                     epochs=epochs, batchsize=batchsize, logdir="", modality=modality,
		                                     initnet=initnet, use3D=use3D, freeze_all=freeze_all, nofreeze=nofreeze,
		                                     extra_epochs=extraepochs, model_version=model_version, mean_path=mean_path,
											 std_path = std_path, with_l2 = with_l2, kinit=kinit,
												 mirror=mirror, augmentation=augmentation, lrmode=lrmode,
												 noise=noise, margintri=margintri, cross_signatures=xs,
												 xsiglevel=xsl,
												 drop_code = dropcode, hdelta = hdelta,
		                                     cameras=cameras, sufix=suffix, 
		                                     alphamob=alphamob, gpool = gpool,
		                                     verbose=verbose, pxk=pxk, k_size=k_size)
		if final_model is not None:
			final_model.save()

	print("* End of training: {}".format(experdir))
