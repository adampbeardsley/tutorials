"""
This simple script takes a photo, fourier transforms it, and creates an animation
where each frame successively adds fourier modes to the image (large scales to small)
A couple libraries are needed: python imaging library (PIL), and progressbar
(comment out if you don't want it). An FFMPEG writer is also required. I used:
conda install -c conda-forge ffmpeg
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Image
import progressbar as pgb

# Some options
nframes = 150
filename = 'DSC_0003.JPG'
fps = 5  # frames per second

# Read in image, FT
raw_im = np.array(Image.open(filename))  # Probably not efficient, but gets us a numpy array to manipulate
raw_im_ft = np.fft.fftshift(np.fft.fft2(raw_im, axes=(0, 1)), axes=(0, 1))
log_power = np.log(np.sum(np.abs(raw_im_ft)**2, axis=2))  # Used to plot FT panel

# Prepare for masking
x0 = raw_im.shape[0] / 2
y0 = raw_im.shape[1] / 2
x = np.arange(raw_im.shape[0]) - x0
y = np.arange(raw_im.shape[1]) - y0
radius = np.sqrt(x.reshape((-1, 1))**2 + y.reshape((1, -1))**2)
rmax = radius.max()
rmin = rmax / 500.0  # This is used because we are going to log bin r for the frames
bin_size = (np.log(rmax) - np.log(rmin)) / nframes
rframes = np.exp(np.log(rmin) + bin_size * (np.arange(nframes) + 1))

# Set up figure for animation
frame = 0
fig, (ax_ft, ax_im) = plt.subplots(1, 2, figsize=(16, 8))
ax_ft.imshow(log_power, animated=True)
foo = ax_ft.axis('off')
ax_im.imshow(raw_im, animated=True)
foo = ax_im.axis('off')


def reset():
    global log_power, raw_im_ft, frame, bar
    frame = 0

    masked_log_power = np.zeros_like(log_power)
    masked_im = np.zeros_like(raw_im_ft, dtype='uint8')

    ax_ft.images[0].set_array(masked_log_power)
    ax_im.images[0].set_array(masked_im)

    bar = pgb.ProgressBar(widgets=[pgb.Percentage(),
                          pgb.Bar(marker='-', left=' |', right='| '), pgb.Counter(),
                          '/{0:0d} frames '.format(nframes),
                          pgb.ETA()], maxval=nframes).start()

    return ax_ft.images[0], ax_im.images[0]


def updatefig(*args):
    global log_power, raw_im_ft, radius, rframes, frame, bar
    frame += 1
    r_ind = radius < rframes[frame]
    masked_ft = raw_im_ft * r_ind[:, :, None]
    masked_log_power = log_power * r_ind
    masked_im = np.real(np.fft.ifft2(np.fft.ifftshift(masked_ft, axes=(0, 1)), axes=(0, 1)))
    masked_im -= masked_im.min()
    masked_im *= 255 / masked_im.max()  # fits into image range
    masked_im = np.uint8(np.round(masked_im))

    ax_ft.images[0].set_array(masked_log_power)
    ax_im.images[0].set_array(masked_im)

    bar.update(frame)

    return ax_ft.images[0], ax_im.images[0]


ani = animation.FuncAnimation(fig, updatefig, repeat=False, frames=nframes - 1,
                              interval=np.round(1.0 / fps * 1000), blit=False, init_func=reset)

# Write the animation
Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
ani.save('animation.mp4', writer=writer)

bar.finish()
