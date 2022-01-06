'''
Based on the following example:
   https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
(c) MJMJ/2020
'''

__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'March 2020'

import os
import numpy as np
import deepdish as dd
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from operator import itemgetter
import copy
import gc
import random
from random import gauss


class DataGeneratorCopyNet(keras.utils.Sequence):
	"""
	A class used to generate data for training/testing CNN gait nets

	Attributes
	----------
	dim : tuple
		dimensions of the input data
	n_classes : int
		number of classes
	...
	"""

	def __init__(self, dataset_info, batch_size=128, mode='train', balanced_classes=True, use3D=False, labmap=[],
				 modality='gray',
				 camera=None, datadir="/home/GAIT_local/TUM_GAID_tf/tfimdb_tum_gaid_N150_train_gray25_60x60/",
				 targets=None, augmentation=True,
				 lstm=False, nframes=None, keep_data=False, mirror=False, mean_sample=0, std_sample=1,
				 multiloss=False, noise=0, nids=150, cross_signatures=False, pxk=False, k=4):
		'Initialization'
		self.use3D = use3D
		self.balanced = balanced_classes
		self.camera = camera
		self.datadir = datadir
		self.empty_files = []
		self.batches = []
		self.modality = modality
		self.mode = mode
		self.keep_data = keep_data
		self.mirror = mirror
		self.data = {}
		self.targets = targets
		self.targets_dim = len(self.targets[list(self.targets.keys())[0]])
		self.mean_sample = mean_sample
		self.std_sample = std_sample
		self.multiloss = multiloss
		self.noise = noise
		self.nids = nids

		if mode == 'train':
			self.set = 1
			self.augmentation = True
		elif mode == 'val':
			self.set = 2
			self.augmentation = False
		elif mode == 'trainval':
			self.set = -1
			self.augmentation = augmentation
		else:
			self.set = 3
			self.augmentation = False

		if mode != 'trainval':
			pos = np.where(dataset_info['set'] == self.set)[0]
			self.gait = dataset_info['gait'][pos]
			self.file_list = list(itemgetter(*list(pos))(dataset_info['file']))
			self.labels = dataset_info['label'][pos]
			self.videoId = dataset_info['videoId'][pos]
			self.indexes = np.arange(len(pos))
		else:
			self.gait = dataset_info['gait']
			self.file_list = dataset_info['file']
			self.labels = dataset_info['label']
			self.videoId = dataset_info['videoId']
			self.indexes = np.arange(len(self.labels))

		nclasses = nids  # len(np.unique(self.labels))    # mjmarin
		if nclasses == 150 or nclasses == 155 or nclasses == 16:
			self.camera = None
			self.cameras = None
		else:
			if "cam" in dataset_info.keys():
				self.cameras = dataset_info['cam']
			else:
				self.cameras = self.__derive_camera(dataset_info)

		if modality != 'silhouette':
			self.__remove_empty_files()
		if self.camera is not None:
			self.__keep_camera_samples()

		self.batch_size = np.min((batch_size, len(self.file_list)))  # Deal with less number of samples than batch size

		self.compressFactor = dataset_info['compressFactor']
		if self.modality == 'of':
			self.dim = (50, 60, 60)
			self.withof = True
		elif self.modality == 'rgb':
			self.dim = (75, 60, 60)
			self.withof = False
		else:
			if self.use3D:
				self.dim = (25, 60, 60, 1)
			else:
				self.dim = (25, 60, 60)
			self.withof = False
		self.ugaits = np.unique(self.gait)
		self.labmap = labmap
		self.list_IDs = []
		self.lstm = lstm
		self.nframes = nframes

		self.__prepare_file_list()

		self.cross_signatures = cross_signatures
		if isinstance(cross_signatures, bool):
			self.cross_signatures_level = 0
		else:
			self.cross_signatures_level = cross_signatures
			self.cross_signatures = True
		if cross_signatures:
			self.__prepare_lab2signatures__()

		self.pxk = pxk
		self.k = k

		if self.pxk:
			self.pxk_prepare()
		self.on_epoch_end()

		self.img_gen = self.__transgenerator(isof=True)

	def __len__(self):
		'Number of batches per epoch'
		return int(np.ceil(len(self.indexes) / np.float(self.batch_size)))

	def __prepare_lab2signatures__(self):
		'Creates dictionary where keys are the labels and content is a list of filenames of samples'
		self.ulabs = np.unique(self.labels)

		self.lab2signatures = {}
		for lix in self.ulabs:
			this_labix = np.where(self.labels == lix)[0]
			self.lab2signatures[lix] = list(itemgetter(*list(this_labix))(self.file_list))

	def __getitem__(self, index):
		"""Generate one batch of data"""
		# Generate data
		X, y = self.__data_generation(self.batches[index])

		if self.lstm:
			if self.modality == 'of':
				# Shape [batch X 2 60 60]
				new_X = np.zeros((X.shape[0], int(X.shape[1] / 2), 2, X.shape[2], X.shape[3]), dtype=X.dtype)
				new_X[:, :, 0, :, :] = X[:, ::2, :, :]
				new_X[:, :, 1, :, :] = X[:, 1::2, :, :]
			else:
				# Shape [batch X 1 60 60]
				new_X = np.zeros((X.shape[0], X.shape[1], 1, X.shape[2], X.shape[3]), dtype=X.dtype)
				new_X[:, :, 0, :, :] = X

			# Build random length samples.
			l = np.random.randint(low=1, high=np.ceil(new_X.shape[1] / 2.0), size=1)[0]  # Number of frames
			i_pos = int(new_X.shape[1] / 2) - l
			e_pos = int(new_X.shape[1] / 2) + l + 1
			X = new_X[:, i_pos:e_pos, :, :, :]

		return X, y

	def __getitemvideoid__(self, index, kindlabel='default'):
		"""Generate one batch of data"""
		# Generate data
		X, y = self.__data_generation(self.batches[index], kindlabel)
		videoId = np.asarray(self.videoId)[self.batches[index]]

		if self.lstm:
			if self.modality == 'of':
				# Shape [batch X 2 60 60]
				new_X = np.zeros((X.shape[0], int(X.shape[1] / 2), 2, X.shape[2], X.shape[3]), dtype=X.dtype)
				new_X[:, :, 0, :, :] = X[:, ::2, :, :]
				new_X[:, :, 1, :, :] = X[:, 1::2, :, :]
			else:
				# Shape [batch X 1 60 60]
				new_X = np.zeros((X.shape[0], X.shape[1], 1, X.shape[2], X.shape[3]), dtype=X.dtype)
				new_X[:, :, 0, :, :] = X

			# Build given length samples.
			if self.nframes is not None:
				i_pos = int(new_X.shape[1] / 2) - self.nframes
				e_pos = int(new_X.shape[1] / 2) + self.nframes + 1
				X = new_X[:, i_pos:e_pos, :, :, :]

		if self.cameras is not None:
			cameras = np.asarray(self.cameras)[self.batches[index]]
		else:
			cameras = None

		return X, y, videoId, cameras

	def __derive_camera(self, dataset_info):
		cameras = []
		for file in dataset_info["file"]:
			# 03314-01-015-01.h5
			parts = file.split("-")
			cameras.append(int(parts[2]))

		return cameras

	def __prepare_file_list(self):
		for g in range(len(self.ugaits)):
			pos = np.where(self.gait == self.ugaits[g])[0]
			self.list_IDs.append(pos)

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		used_samples = []
		free_position = []

		if self.pxk:
			self.on_epoch_end_pxk()
			tf.keras.backend.clear_session()
			gc.collect()
			return
		else:
			for g in range(len(self.ugaits)):
				np.random.shuffle(self.list_IDs[g])
				used_samples.append(np.zeros(len(self.list_IDs[g])))
				free_position.append(0)

		self.batches = []

		samples_per_gait = int(np.floor(self.batch_size / len(self.ugaits)))
		for i in range(self.__len__()):
			self.batches.append([])
			for g in range(len(self.ugaits)):
				in_pos = free_position[g]
				en_pos = in_pos + samples_per_gait

				if en_pos >= len(self.list_IDs[g]):
					# We take the last samples and shuffle again.
					self.batches[i].extend(self.list_IDs[g][in_pos:len(self.list_IDs[g])])
					np.random.shuffle(self.list_IDs[g])

					if self.mode != 'test':
						new_en_pos = samples_per_gait - (len(self.list_IDs[g]) - in_pos)
						self.batches[i].extend(self.list_IDs[g][0:new_en_pos])
						used_samples[g] = np.zeros(len(used_samples[g]))
						used_samples[g][0:new_en_pos] = 1
						free_position[g] = new_en_pos
				else:
					self.batches[i].extend(self.list_IDs[g][in_pos:en_pos])
					used_samples[g][in_pos:en_pos] = 1
					free_position[g] = en_pos

			# The remaining samples are selected randomly from the different kind of gaits.
			if self.mode != 'test':
				rem = int(self.batch_size - len(self.batches[i]))
				gait_temp = np.arange(0, len(self.ugaits))
				np.random.shuffle(gait_temp)
				for j in range(rem):
					self.batches[i].extend([self.list_IDs[gait_temp[j]][free_position[gait_temp[j]]]])
					used_samples[gait_temp[j]][free_position[gait_temp[j]]] = 1
					free_position[gait_temp[j]] += 1

					# If all samples are used, we must shuffle again...
					if free_position[gait_temp[j]] >= len(self.list_IDs[gait_temp[j]]):
						np.random.shuffle(self.list_IDs[gait_temp[j]])
						used_samples[gait_temp[j]] = np.zeros(len(used_samples[gait_temp[j]]))
						free_position[gait_temp[j]] = 0
			self.batches[i] = np.asarray(self.batches[i])

		tf.keras.backend.clear_session()
		gc.collect()

	def __load_dd(self, filepath: str, clip_max=0, clip_min=0):
		"""
		Loads a dd file with gait samples
		:param filepath: full path to h5 file (deep dish compatible)
		:return: numpy array with data
		"""
		if filepath is None or not os.path.exists(filepath):
			return None

		if self.keep_data:
			if filepath in self.data:
				sample = self.data[filepath]
			else:
				sample = dd.io.load(filepath)
				self.data[filepath] = copy.deepcopy(sample)
		else:
			sample = dd.io.load(filepath)

		if len(sample["data"]) == 0:
			return None

		if sample["compressFactor"] > 1:
			x = np.float32(sample["data"])
			# import pdb; pdb.set_trace()
			if clip_max > 0:
				x[np.abs(x) > clip_max] = 1e-8
			if clip_min > 0:
				x[np.abs(x) < clip_min] = 1e-8
			x = x / sample["compressFactor"]

		# x = x * 0.1  # DEVELOP!
		else:
			if self.modality == 'gray':
				x = (np.float32(sample["data"]) / 255.0) - 0.5
			elif self.modality == 'silhouette':
				x = np.float32(sample["data"]) / 255.0
			elif self.modality == 'depth':
				# import pdb; pdb.set_trace()
				x = (np.float32(sample["data"]) / 255.0) - 0.5

		x = np.moveaxis(x, 2, 0)

		if not (type(self.mean_sample) is int) or self.mean_sample != 0:
			x = x - self.mean_sample

		if not (type(self.std_sample) is int) or self.std_sample != 1:
			x = x / self.std_sample

		return x

	def __remove_empty_files(self):
		gait_ = []
		file_list_ = []
		labels_ = []
		videoId_ = []
		indexes_ = []
		for i in range(len(self.file_list)):
			datafile = os.path.join(self.datadir, self.file_list[i])
			if not os.path.exists(datafile):  # TODO Warning FIXME DEVELOP!!
				print("WARN: missing " + datafile)
				continue
			data_i = dd.io.load(datafile)
			if len(data_i["data"]) > 0:
				gait_.append(self.gait[i])
				file_list_.append(self.file_list[i])
				labels_.append(self.labels[i])
				videoId_.append(self.videoId[i])
				indexes_.append(self.indexes[i])

		self.gait = gait_
		self.file_list = file_list_
		self.labels = labels_
		self.videoId = videoId_
		self.indexes = indexes_

	def __keep_camera_samples(self):
		gait_ = []
		file_list_ = []
		labels_ = []
		videoId_ = []
		indexes_ = []
		for i in range(len(self.file_list)):
			for j in range(len(self.camera)):
				cam_str = "{:03d}".format(self.camera[j])
				if cam_str in self.file_list[i]:
					gait_.append(self.gait[i])
					file_list_.append(self.file_list[i])
					labels_.append(self.labels[i])
					videoId_.append(self.videoId[i])
					indexes_.append(self.indexes[i])

		self.gait = gait_
		self.file_list = file_list_
		self.labels = labels_
		self.videoId = videoId_
		self.indexes = indexes_

	def __gen_batch(self, list_IDs_temp, kindlabel='default'):
		# Initialization
		# if self.use3D:
		dim0 = len(list_IDs_temp)
		x = np.empty((dim0, *self.dim))
		if kindlabel == 'default':
			if self.multiloss:
				y = [np.empty((dim0, self.targets_dim)), np.empty((dim0, 1)), np.empty((dim0, 1))]
			else:
				y = np.empty((dim0, self.targets_dim))
		else:
			y = np.empty((dim0, 1))

		# Generate data
		for i, ID in enumerate(list_IDs_temp):

			filepath = os.path.join(self.datadir, self.file_list[ID])
			label = self.labels[ID]

			# Data augmentation?
			if self.augmentation and np.random.randint(4) > 0:
				trans = self.img_gen.get_random_transform((self.dim[1], self.dim[2]))
				flip = self.mirror and np.random.randint(2) == 1  # mjmarin: flip is controlled by 'self.mirror'
			else:
				trans = None
				flip = False

			if self.withof and self.augmentation and np.random.randint(2) == 1:
				clip_max = int(gauss(2300, 10))  # 2300
				clip_min = int(gauss(50, 3))  # 50
			else:
				clip_max = 0
				clip_min = 0

			# Store sample
			x_tmp = self.__load_dd(filepath, clip_max=clip_max, clip_min=clip_min)

			if x_tmp is None:
				print("WARN: this shouldn't happen!")
				import pdb
				pdb.set_trace()

			else:
				if trans is not None:
					x_tmp = self.__transformsequence(x_tmp, self.img_gen, trans)
					if flip:
						x_tmp = self.__mirrorsequence(x_tmp, self.withof, True)
				elif self.mirror and self.mode != 'test':
					x_tmp = self.__mirrorsequence(x_tmp, self.withof, True)

				#				if self.mirror and flip and self.mode == 'trainval':
				#					x_tmp = self.__mirrorsequence(x_tmp, self.withof, True)

				if self.use3D and x_tmp is not None:
					if self.dim[0] == 25:  # Gray or Depth
						x_tmp = np.expand_dims(x_tmp, axis=3)
					elif self.dim[0] == 50:  # OF
						x_temp_ = np.zeros((25, self.dim[1], self.dim[2], 2), dtype=np.float32)
						for i_xtmp in range(25):
							in_p = i_xtmp * 2
							en_p = in_p + 2
							x_temp_[i_xtmp, :, :, :] = x_tmp[in_p:en_p, :, :]

						x_tmp = x_temp_
					elif self.dim[0] == 75:  # RGB
						x_temp_ = np.zeros((25, self.dim[1], self.dim[2], 3), dtype=np.float32)
						for i_xtmp in range(25):
							in_p = i_xtmp * 3
							en_p = in_p + 3
							x_temp_[i_xtmp, :, :, :] = x_tmp[in_p:en_p, :, :]

						x_tmp = x_temp_

				x[i] = x_tmp

			# if self.file_list[ID] in self.targets:
			if self.cross_signatures and np.random.randint(4) > self.cross_signatures_level:
				IDx_rnd = random.randint(0, len(self.lab2signatures[label]) - 1)
				keyname = self.lab2signatures[label][IDx_rnd]
			else:
				keyname = self.file_list[ID]

			if self.mirror and flip and self.mode != 'test':  # self.mirror and self.mode != 'test':
				keyname = keyname + "-mirror"

			the_target = self.targets[keyname]
			if self.noise > 0:
				the_target = the_target + self.noise * 0.0001 * np.random.randn(len(the_target))  # DEVELOP: 0.0001

			if kindlabel == 'default':
				if self.multiloss:
					y[0][i,] = the_target
					if self.labmap:
						y[1][i] = self.labmap[int(label)]
					else:
						y[1][i] = label
				else:
					y[i,] = the_target
			else:
				if self.labmap:
					y[i,] = self.labmap[int(label)]
				else:
					y[i,] = label
		# else:
		#	y[i,] = np.random.rand(self.targets_dim) -0.5
		#	print(":: Key error: "+self.file_list[ID])

		if self.multiloss:
			y[2] = y[1]

		return x, y

	def __data_generation(self, list_IDs_temp, kindlabel='default'):
		'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
		x, y = self.__gen_batch(list_IDs_temp, kindlabel)

		return x, y

	def __mirrorsequence(self, sample, isof=True, copy=True):
		"""
		Returns a new variable (previously copied), not in-place!
		:rtype: numpy.array
		:param sample:
		:param isof: boolean. If True, sign of x-channel is changed (i.e. direction changes)
		:return: mirror sample
		"""
		# Make a copy
		if copy:
			newsample = np.copy(sample)
		else:
			newsample = sample

		nt = newsample.shape[0]
		for i in range(nt):
			newsample[i,] = np.fliplr(newsample[i,])
			if i % 2 == 0 and isof:
				newsample[i,] = -newsample[i,]

		return newsample

	def __transformsequence(self, sample, img_gen, transformation):
		sample_out = np.zeros_like(sample)
		# min_v, max_v = (sample.min(), sample.max())
		abs_max = np.abs(sample).max()
		for i in range(sample.shape[0]):
			I = np.copy(sample[i,])
			I = np.expand_dims(I, axis=2)
			It = img_gen.apply_transform(I, transformation)

			sample_out[i,] = It[:, :, 0]

		# Fix range if needed
		if np.abs(sample_out).max() > (3 * abs_max) and self.modality != 'silhouette':  # This has to be normalized
			sample_out = (sample_out / 255.0) - 0.5

		return sample_out

	def __transgenerator(self, displace=[-5, -3, 0, 3, 5], isof=True):

		if isof:
			ch_sh_range = 0
			br_range = None
		else:
			ch_sh_range = 0.025
			br_range = [0.95, 1.05]

		img_gen = ImageDataGenerator(width_shift_range=displace, height_shift_range=displace,
									 brightness_range=br_range, zoom_range=0.04,
									 channel_shift_range=ch_sh_range, horizontal_flip=False)

		return img_gen

	def pxk_prepare(self):
		ids = []
		for i in range(len(np.unique(self.labels))):
			label_ = np.unique(self.labels)[i]
			id_gait = []
			for j in range(len(self.ugaits)):
				id_gait.append([self.list_IDs[j][k] for k in range(len(self.list_IDs[j])) if self.labels[self.list_IDs[j][k]] == label_])
				np.random.shuffle(id_gait[j])
			ids.append(id_gait)
		self.list_IDs_pxk = ids


	def on_epoch_end_pxk(self):

		self.batches = []
		id_counter = 0
		gait_counter = 0
		labels_ = np.unique(self.labels)
		np.random.shuffle(labels_)
		counter = np.zeros(shape=np.shape(self.list_IDs_pxk), dtype=np.int)
		for i in range(self.__len__()):
			self.batches.append([])
			for j in range(int(self.batch_size / self.k)):
				id_idx = np.where(np.unique(self.labels) == labels_[id_counter])[0][0]
				for k in range(self.k):
					self.batches[i].append(self.list_IDs_pxk[id_idx][gait_counter][counter[id_idx][gait_counter]])
					counter[id_idx][gait_counter] += 1
					if counter[id_idx][gait_counter] >= len(self.list_IDs_pxk[id_idx][gait_counter]):
						counter[id_idx][gait_counter] = 0
					gait_counter += 1
					if gait_counter >= len(self.ugaits):
						gait_counter = 0
				id_counter += 1
				if id_counter >= len(labels_):
					id_counter = 0
					np.random.shuffle(labels_)