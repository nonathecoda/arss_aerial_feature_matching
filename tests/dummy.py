import h5py
from icecream import ic
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import nexusformat.nexus as nx
import h5py

import matplotlib.pyplot as plt
import cv2
import numpy as np

file_labels_training = "/Users/antonia/dev/UNITN/remote_sensing_systems/arss_aerial_feature_matching/groundtruth_gui/labels_training.hdf5"

with h5py.File(file_labels_training, 'r') as f:
        #print(f.keys())
        ic(f.keys())
        keyzero = list(f.keys())[0]
        print(f[keyzero].keys())
        #ic(list(f[keyzero]['keypoints']))
        
        

file_training = "/Users/antonia/dev/UNITN/remote_sensing_systems/arss_aerial_feature_matching/groundtruth_gui/training.hdf5"

with h5py.File(file_training, 'r') as f:
        ic(f.keys())
        keyzero = list(f.keys())[0]
        ic(f[keyzero].keys())
        
        img = np.hstack((np.array(f[keyzero]['optical']), np.array(f[keyzero]['thermal'])))
        '''while(1):
            cv2.imshow('Image Pair with Keypoints', img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        '''


exit()

# Load the images
cam1 = "/Users/antonia/Desktop/Screenshot 2024-01-14 at 15.29.24.png"
img_optical = cv2.imread(cam1, 0)
cam2 = "/Users/antonia/Desktop/Screenshot 2024-01-14 at 15.30.38.png"
img_thermal = cv2.imread(cam2, 0)

#resize images to 233, 314
dog = cv2.resize(img_optical, (312, 232))
cat = cv2.resize(img_thermal, (312, 232))
'''
#normalize thermal images (you could also use np.clip)
img_thermal = np.clip(img_thermal, np.percentile(img_thermal, 1), np.percentile(img_thermal, 99))
dog = img_thermal/img_thermal.max()
# normalize optical images
cat = img_optical/img_optical.max()
'''
overlay_mask = np.zeros(cat.shape[:2], dtype=np.uint8)
overlay_alpha = 0.8
blended_image = cv2.addWeighted(cat, overlay_alpha, dog, 1, 0)
#fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#ax[0].imshow(dog, cmap='gray')
#ax[1].imshow(cat, cmap='gray')
#ax[0].imshow(dog, cmap='gray')
#ax[1].imshow(cat, cmap='gray')


def on_key_press(event):
    if event.key == 'q':
        plt.close()  # Close the window when 'q' is pressed
    elif event.key == 'p':
        print("Hello world")  # Print "Hello world" when 'p' is pressed

plt.imshow(cat, cmap='gray')
plt.imshow(dog, alpha=0.5, cmap='gray')
plt.connect('key_press_event', on_key_press)
plt.show()

exit()
while(1):
        cv2.imshow('Image Pair with Keypoints', cat)
        if cv2.waitKey(20) & 0xFF == 27:
                print("hello")
        elif cv2.waitKey(20) & 0xFF == 32:
                break
cv2.destroyAllWindows()