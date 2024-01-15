from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
from icecream import ic
from scipy.optimize import minimize

def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

'''def compute_mi_score(img1, img2, H):
    """Computes the mutual information (MI) score between two images warped according to a homography H."""
    M = H.reshape(3,3)
    warped_img1 = cv2.warpPerspective(img1, M, dsize=(img2.shape[1], img2.shape[0]), flags=cv2.INTER_CUBIC)
    
    # Compute histograms of the images
    hist_2d, x_edges, y_edges = np.histogram2d(warped_img1.ravel(), img2.ravel(),bins=[64, 32])
    mi_score = mutual_information(hist_2d)
    
    return mi_score
'''

def compute_mi_score(img1, img2, H):
    """Computes the mutual information (MI) score between two images warped according to a homography H."""
    
    #M = H.reshape(3,3)
    warped_img1 = cv2.warpPerspective(img1, H.reshape(3,3), dsize=(img2.shape[1], img2.shape[0]), flags=cv2.INTER_CUBIC)
    # Compute histograms of the images
    hist1 = np.histogram(img2.ravel(), bins=256)[0]
    hist2 = np.histogram(warped_img1.ravel(), bins=256)[0]
    
    # Compute the MI score
    # Compute histograms of the images
    hist_2d, x_edges, y_edges = np.histogram2d(warped_img1.ravel(), img2.ravel(),bins=[400, 200])
    mi_score = mutual_information(hist_2d)
    '''
    hist_2d_moved_log = np.zeros(hist_2d.shape)
    non_zeros = hist_2d != 0
    hist_2d_moved_log[non_zeros] = np.log(hist_2d[non_zeros])
    plt.imshow(hist_2d_moved_log)
    plt.show()
    
    ic(H)
    ic(mi_score)
    exit()
    '''
    ic(H)
    ic(mi_score)
    return mi_score

def optimize_homography(img1, img2):
    """Optimizes the homography between images img1 and img2 using the MI score as the cost function."""
    
    H = np.eye(3); H.flatten()
    ic(H.ndim)

    # Define the cost function
    def cost_function(H):
        return -compute_mi_score(img1, img2, H)
    
    # Perform optimization
    res = minimize(cost_function, H, method='nelder-mead',options={'xatol': 1e-8, 'disp': True, 'maxiter': 1000})
    optimized_homography = res.x

    return optimized_homography


'''
step 0: load images
'''

cam1 = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam1/cam1_00138_e_0010_g_01_87968505_corr_rect.tiff"
camT = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/camT/camT_00130_00086269_rect.tiff"

img_optical = cv2.imread(cam1, 0)
img_thermal = cv2.imread(camT, 0)

'''
step 1: normalize images
'''

# normalize and resize optical images
img_optical = cv2.resize(img_optical, (img_thermal.shape[1], img_thermal.shape[0]))
img_optical_n = img_optical/img_optical.max()

#normalize thermal images
img_thermal[img_thermal > np.percentile(img_thermal, 99)] = np.percentile(img_thermal, 99)
img_thermal[img_thermal < np.percentile(img_thermal, 1)] = np.percentile(img_thermal, 1)
img_thermal_n = img_thermal/img_thermal.max()
'''
step2: Find homography from optical to thermal image
    - use nelder-mead optimizer to maximise MI score
    - MI score: https://matthew-brett.github.io/teaching/mutual_information.html
'''


homography = optimize_homography(img_optical_n, img_thermal_n).reshape(3,3)
ic(homography)

'''
plot images
'''

img_optical_warped = cv2.warpPerspective(img_optical_n, homography, dsize=(img_thermal_n.shape[1], img_thermal_n.shape[0]), flags=cv2.INTER_CUBIC)

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(img_optical_n, cmap = 'gray')
axs[0, 0].set_title('img_optical_n')
axs[0, 1].imshow(img_optical_warped, cmap = 'gray')
axs[0, 1].set_title('img_optical_warped')
axs[1, 0].imshow(img_thermal_n, cmap = 'gray')
axs[1, 0].set_title('img_thermal_n')
axs[1, 1].imshow(img_thermal_n, cmap = 'gray')
axs[1, 1].set_title('img_thermal_n')
plt.show()
