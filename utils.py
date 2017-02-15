import cv2
import math
import matplotlib.pyplot as plt

class imageUtils(object):
	"""utility class for image processing """
	
	def __init__(self):
		pass

	# Final image size 32x32x3 to be used by the CNN
	im_x = 32
	im_y = 32
	im_z = 3

	def plot_images(self, images):
		""" Function to plot test images """
		for index in range(len(images)):
		    plt.imshow(images[index])
		plt.show()

	def pre_process_image(self, img):
		"""Function to pre process images before feeding them into the network """
		
		#self.plot_images([img])
		# Get image shape
		img_shape = img.shape
		# Crop the top and bottom to remove unwanted features
		img = img[math.floor(img_shape[0]*.40):math.floor(img_shape[0]*.85), 0:img.shape[1]]
		# Normalize the image to mitigate differences due to light condition across the data set
		# and will make the pixel intesity consistant. Also useful for different tracks.
		img = cv2.normalize(img, img, 40, 100, cv2.NORM_MINMAX)
		# Convert to HSV color space. This seems to work better compare to RGB, BGR and GRAY
		# and is also easier to normalize the value channel. 
		img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		# Resize the image to 32x32x3 for CNN 
		img = cv2.resize(img,(self.im_x,self.im_y), interpolation=cv2.INTER_AREA)
		#self.plot_images([img])
		return img
