import numpy as np
from random import uniform
from skimage.color import rgb2hsv,hsv2rgb
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img


# Covert a set of images from RGB to HSV colour space
def rgb2hsv_set(img_set):

	hsv_img_set = np.ndarray(shape=img_set.shape, dtype=np.float32)

	i = 0
	for rgb_img in img_set:
		hsv_img = rgb2hsv(rgb_img)
		hsv_img_set[i] = hsv_img
		i = i+1

	return hsv_img_set

# Covert a set of images from HSV to RGB colour space
def hsv2rgb_set(img_set):

	rgb_img_set = np.ndarray(shape=img_set.shape, dtype=np.float32)

	i = 0
	for hsv_img in img_set:
		rgb_img = hsv2rgb(hsv_img)
		rgb_img_set[i] = rgb_img
		i = i+1

	return rgb_img_set

# Illuminate a set of HSV images using gamma correction
def illuminate_images(img_set):
	# Image parameters
	img_width = 100
	img_height = img_width
	img_channels = 3
	illuminated_set = np.ndarray(shape=img_set.shape, dtype=np.float32)

	i = 0
	for data in img_set:
		img =  data.copy()

		# Create random adjusting parameter
		illumination = 1 / uniform(1.6, 3.3)

		# Apply gamma correction to enhance illumination
		for k in range(img_width):
			for j in range(img_height):
				img[k,j,2] = pow(img[k,j,2], illumination)

		illuminated_set[i] = img
		i = i+1

	return illuminated_set

# Add random noise to V channel on a set of HSV images
def add_noise(img_set):
	# Image parameters
	img_width = 100
	img_height = img_width
	img_channels = 3
	noisy_set = np.ndarray(shape=img_set.shape, dtype=np.float32)

	# Create random Gaussian distributed noise
	noise_factor = 0.01
	mean = 0.0
	stand_dev = uniform(10, 18)

	noise =  np.random.normal(loc=mean,
				scale=stand_dev,
				size=(len(img_set), img_width, img_height))

	for i in range(len(img_set)):
		noisy_set[i,:,:,0] = img_set[i,:,:,0]
		noisy_set[i,:,:,1] = img_set[i,:,:,1]
		noisy_set[i,:,:,2] = img_set[i,:,:,2].copy() + (noise[i,:,:] * noise_factor)

	# Keep pixel values in range [0,1]
	noisy_set = np.clip(noisy_set, 0., 1.)

	return noisy_set

# Reduce a set of images to 1 colour channel
def reduce_to_1(img_set, channel):
	# Image parameters
	img_width = 100
	img_height = img_width
	img_channels = 3
	reduced_set = np.ndarray(shape=(len(img_set), img_width, img_height, 1),
	                    	dtype=np.float32)

	i = 0
	for img in img_set:
		reduced_set[i,:,:,0] = img[:,:,channel]
		i = i+1

	return reduced_set

# Update the value channel in a set of HSV images
def update_V(img_set, V_set):

	updated_set = np.ndarray(shape=img_set.shape, dtype=np.float32)

	for i in range(len(img_set)):
		updated_set[i,:,:,0] = img_set[i,:,:,0]
		updated_set[i,:,:,1] = img_set[i,:,:,1]
		updated_set[i,:,:,2] = V_set[i,:,:]

	return updated_set

# Reduce a set of HSV images to just the value channel
def reduce_to_V(img_set):

	reduced_set = np.ndarray(shape=img_set.shape, dtype=np.float32)

	i = 0
	for img in img_set:
		reduced_set[i,:,:,0] = 0
		reduced_set[i,:,:,1] = 0
		reduced_set[i,:,:,2] = img[:,:,2]
		i = i+1

	return reduced_set

# Create a plot of images
def create_plot(rows, cols, data, titles, size):

	figsize = [size, size]
	fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

	for j in range(rows):
		arr = np.asarray(data[j])
		ax[j,2].set_title(titles[j])

		for i in range(cols):
			img = array_to_img(arr[i])
			ax[j,i].imshow(img)
			ax[j,i].axis('off')

	return fig
