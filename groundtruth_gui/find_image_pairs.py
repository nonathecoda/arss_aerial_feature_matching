from icecream import ic
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time

class GUI:

    def __init__(self):

        self.path_to_cam_1_folder = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam1"
        self.path_to_cam_T_folder = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/camT"

        current_time = time.strftime("_%H_%M_%S", time.localtime())
        chosen_pairs = '/Users/antonia/dev/UNITN/remote_sensing_systems/arss_aerial_feature_matching/groundtruth_gui/data/pairs/datapairs' + str(current_time) + '.txt'
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
                if i > 70 and j < 50:
                    continue
                elif i > 107 and j < 75:
                    continue
                elif i > 130 and j < 95:
                    continue
                elif i > 150 and j < 110:
                    continue
                elif i > 170 and j < 125:
                    continue
                elif i > 180 and j < 140:
                    continue
                elif i > 200 and j < 155:
                    continue
                elif i > 217 and j < 170:
                    continue
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
                f.canvas.manager.set_window_title('n: no, y: yes, o: next optical, x: exit')
                plt.show()

                if self.chosen_key == "y":
                    row_string = str(item1) + ", " + str(itemT)
                    self.pairs_file.write(row_string + '\n')
                    self.pairs_file.flush()
                    print("yes")
                elif self.chosen_key == "n":
                    print("no")
                elif self.chosen_key == "o":
                    print("next optical")
                    break
    
    def choose_homography_key_event(self, event):
        if event.key == 'n':
            self.chosen_key = "n"
            plt.close()  
        elif event.key == 'y':
            self.chosen_key = "y"
            plt.close()  
        elif event.key == 'o':
            self.chosen_key = "o"
            plt.close() 
        elif event.key == 'x':
            plt.close()
            exit()
        





if __name__ == "__main__":
    gui = GUI()
    gui.main()