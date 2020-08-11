# ----------------------------------------------------------------------------------------------------------------------
#   Author: Sean Deely
#   ID:     1674836
#   Date:   20/04/20
# ----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import scipy.fftpack as fft
import numpy as np
from skimage.transform import rotate
from PIL import Image


# ----------------------------------------------------------------------------------------------------------------------
def fft_freq_domain(projections):
    """" Build 1-d FFTs of an array of projections, each projection 1 row of the array. """
    return fft.rfft(projections, axis=1)
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def ramp_filter(ffts):
    """ Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows). """
    ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5))
    return ffts * ramp
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def irfft_spatial_domain(filtered_projections):
    """" Build 1-d IRFFTs of an array of projections, each projection 1 row of the array. """
    return fft.irfft(filtered_projections, axis=1)
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def back_project(filtered_projections):
    """ Generate a laminogram by simple backprojection using the Radon Transform of an image."""
    laminogram = np.zeros((filtered_projections.shape[1], filtered_projections.shape[1]))
    dTheta = 180.0 / filtered_projections.shape[0]
    for i in range(filtered_projections.shape[0]):
        temp = np.tile(filtered_projections[i], (filtered_projections.shape[1], 1))
        temp = rotate(temp, dTheta * i)
        laminogram += temp
    return laminogram
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def crop_image(img, crop_x, crop_y):
    """ This function crops the center portion of the reconstructed
        image, removing any unwanted borders which may exist. """
    width = img.shape[0]
    height = img.shape[1]
    start_x = width//2-(crop_x//2)
    start_y = height//2-(crop_y//2)
    return img[start_y:start_y + crop_y, start_x:start_x + crop_x]
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def reconstruct_channel_nf(channel):
    """ Reconstructs each channel of the image with no ramp-filtering. """
    unfiltered_reconstruction = back_project(channel)
    cropped_channel = crop_image(unfiltered_reconstruction, 767, 575)
    return (255 * (cropped_channel - np.min(cropped_channel))
            / np.ptp(cropped_channel)).astype('uint8')
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def reconstruct_channel_wf(channel):
    """ Reconstructs each channel of the image with ramp-filtering. """
    freq_domain_projections = fft_freq_domain(channel)
    filtered_freq_domain_projections = ramp_filter(freq_domain_projections)
    filtered_spatial_domain_projections = irfft_spatial_domain(filtered_freq_domain_projections)
    cropped_channel = back_project(filtered_spatial_domain_projections)
    # cropped_channel = crop_image(reconstructed_channel, 767, 575)
    return (255 * (cropped_channel - np.min(cropped_channel))
            / np.ptp(cropped_channel)).astype('uint8')
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def reconstruct_channel_hamming(channel):
    """ Reconstructs each channel of the image using hamming-windowed ramp-filtering. """
    freq_domain_projections = fft_freq_domain(channel)
    filtered_freq_domain_projections = ramp_filter(freq_domain_projections)
    filtered_spatial_domain_projections = irfft_spatial_domain(filtered_freq_domain_projections)
    reconstructed_channel = back_project(filtered_spatial_domain_projections)

    hamming_window = np.hamming(reconstructed_channel.shape[0])
    hamming_reconstruction = reconstructed_channel * hamming_window
    # hanning_window = np.hanning(reconstructed_channel.shape[0])
    # hanning_reconstruction = reconstructed_channel * hanning_window

    cropped_channel = crop_image(hamming_reconstruction, 767, 575)
    return (255 * (cropped_channel - np.min(cropped_channel))
            / np.ptp(cropped_channel)).astype('uint8')
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    sinogram = Image.open("sinogram.png")

    r_channel = np.array(sinogram.getchannel('R'))
    g_channel = np.array(sinogram.getchannel('G'))
    b_channel = np.array(sinogram.getchannel('B'))

    original_channels = [r_channel,g_channel,b_channel]

    reconstructed_channels = []
    for channel in original_channels:
        reconstructed_channels.append(reconstruct_channel_wf(channel))
        # reconstructed_channels.append(reconstruct_channel_nf(channel))
        # reconstructed_channels.append(reconstruct_channel_hamming(channel))

    laminogram = np.dstack(tuple(reconstructed_channels))
    plt.imshow(laminogram)
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------
