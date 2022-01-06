# Code for data augmentation, applied to gait sequences
# (c) MJMJ/2020

__author__ = 'Manuel J Marin-Jimenez'
__copyright__ = 'April 2020'

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def mj_mirrorsequence(sample, isof=True, copy=True):
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
        if i % 2 == 0:
            newsample[i,] = -newsample[i,]

    return newsample


def mj_transformsequence(sample, img_gen, transformation):
    sample_out = np.zeros_like(sample)
    abs_max = np.abs(sample).max()
    for i in range(sample.shape[0]):
        I = np.copy(sample[i, ])
        I = np.expand_dims(I, axis=2)
        It = img_gen.apply_transform(I, transformation)

        sample_out[i, ] = It[:, :, 0]

    # Fix range if needed
    if np.abs(sample_out).max() > 3*abs_max: # This has to be normalized
        sample_out = (sample_out /255.0) - 0.5

    return sample_out


def mj_transgenerator(displace=[-5, -3, 0, 3, 5], isof=True):

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
