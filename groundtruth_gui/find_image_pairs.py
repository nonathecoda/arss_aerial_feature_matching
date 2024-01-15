from icecream import ic
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


class GUI:

    def __init__(self):

        self.path_to_cam_1_folder = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam1"
        self.path_to_cam_T_folder = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/camT"

        chosen_pairs = '/Users/antonia/dev/UNITN/remote_sensing_systems/arss_aerial_feature_matching/groundtruth_gui/pairs.txt'
        self.pairs_file = open(chosen_pairs, 'w')

        self.img_optical = None
        self.img_thermal = None
        self.chosen_key = None

        self.items_cam1 = os.listdir(self.path_to_cam_1_folder)
        self.items_cam1.sort() 
        self.items_cam1.pop(0)
        
        self.items_camT = os.listdir(self.path_to_cam_T_folder)
        self.items_camT.sort()
        self.items_camT.pop(0)

    def main(self):
        i = 0
        for item1 in self.items_cam1:
            i += 1
            j = 0
            path_to_cam1_img = os.path.join(self.path_to_cam_1_folder, item1)
            for itemT in self.items_camT:
                j += 1
                ic(item1)
                ic(itemT)

                path_to_camT_img = os.path.join(self.path_to_cam_T_folder, itemT)

                img_optical = cv2.imread(path_to_cam1_img, 0)
                img_thermal = cv2.imread(path_to_camT_img, 0)

                #resize images to 233, 314
                img_optical = cv2.resize(img_optical, (312, 232))
                img_thermal = cv2.resize(img_thermal, (312, 232))

                #normalize thermal images (you could also use np.clip)
                img_thermal = np.clip(img_thermal, np.percentile(img_thermal, 1), np.percentile(img_thermal, 99))
                self.img_thermal = img_thermal/img_thermal.max()
                # normalize optical images
                self.img_optical = img_optical/img_optical.max()

                # plot different homographies
                f, axarr = plt.subplots(1,2, figsize=(20, 10))
                axarr[0].imshow(self.img_optical, cmap='gray')
                axarr[0].set_title('img_optical nr. ' + str(i) + ' /221')
                axarr[1].imshow(self.img_thermal, cmap='gray')
                axarr[1].set_title('img thermal nr. ' + str(j) + ' /305')
                plt.connect('key_press_event', self.choose_homography_key_event)
                plt.show()

                if self.chosen_key == "y":
                    row_string = str(item1) + ", " + str(itemT)
                    self.pairs_file.write(row_string + '\n')
                    self.pairs_file.flush()
                    print("yes")
                elif self.chosen_key == "n":
                    print("no")
                elif self.chosen_key == "q":
                    print("quit")
                    break
    
    def choose_homography_key_event(self, event):
        if event.key == 'n':
            self.chosen_key = "n"
            plt.close()  # Close the window when 'q' is pressed
        elif event.key == 'y':
            self.chosen_key = "y"
            plt.close()  # Close the window when 'q' is pressed
        elif event.key == 'q':
            self.chosen_key = "q"
            plt.close()  # Close the window when 'q' is pressed
        elif event.key == 'x':
            plt.close()
            exit()
        





if __name__ == "__main__":
    gui = GUI()
    gui.main()