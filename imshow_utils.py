import cv2
import matplotlib.pylab as plt

global image
image = None 
def imshow_plt(label, im):
    global image
    if image is None: 
        image = plt.imshow(im, interpolation=None, animated=True, label=label)
    image.set_data(im)
    plt.draw()

def imshow_cv(label, im): 
    cv2.imshow(label, im)
    cv2.waitKey(1)
