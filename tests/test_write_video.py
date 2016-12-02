from pybot.utils.test_utils import test_dataset
from pybot.utils.io_utils import write_video
from pybot.vision.image_utils import to_color

if __name__ == "__main__": 
    dataset = test_dataset(scale=0.5)
    for left_im, right_im in dataset.iter_stereo_frames(): 
        write_video('video.avi', to_color(left_im))
