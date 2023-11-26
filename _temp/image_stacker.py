
# private
# pip install opencv-python-headless natsort

import os
import glob
import shutil 

# import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import natsort
import numpy as np

    
if __name__=="__main__":

    data_root = "/home/ailab-ur5/_Workspaces/_data/uop/20231126_uop_ycb_selected+cropped+resized/ycb/"
    dst_root = "/home/ailab-ur5/_Workspaces/_data/uop/20231126_uop_ycb_stacked/ycb/"


    # list_obj_folder = os.listdir(data_root)
    list_obj_folder = glob.glob(data_root + "*/")
    sorted_list_obj_folder = natsort.natsorted(list_obj_folder)
    
    list_not_maded = []
    for idx, one_obj_folder in tqdm.tqdm(enumerate(sorted_list_obj_folder)):

        if idx < 4:
            continue
        
        list_img_folder = glob.glob(one_obj_folder + "recorded_data/*/")
        sorted_list_img_folder = natsort.natsorted(list_img_folder)

        list_img_file_0 = glob.glob(sorted_list_img_folder[0] + "*.png")    # : label
        sorted_list_img_file_0 = natsort.natsorted(list_img_file_0)

        list_img_file_1 = glob.glob(sorted_list_img_folder[1] + "*.png")    # : partial
        sorted_list_img_file_1 = natsort.natsorted(list_img_file_1)

        list_img_file_2 = glob.glob(sorted_list_img_folder[2] + "*.png")    # : whole
        sorted_list_img_file_2 = natsort.natsorted(list_img_file_2)


        one_img_folder_split = sorted_list_img_file_0[0].split("/")

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.ion()
        # plt.show()

        for idx, one_idx_image_files in enumerate(zip(sorted_list_img_file_2, sorted_list_img_file_0, sorted_list_img_file_1)):
            try:
                # pil_image_0 = Image.open(one_idx_image_files[0]).copy()
                # pil_image_1 = Image.open(one_idx_image_files[1]).copy()
                # pil_image_2 = Image.open(one_idx_image_files[2]).copy()

                imgs = [ Image.open(i) for i in one_idx_image_files ]
                # # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
                # min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
                # imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])
                hstacked_array = np.hstack([ i for i in imgs ])

                hstacked_image = Image.fromarray(hstacked_array)

                save_name = one_obj_folder.replace("20231126_uop_ycb_selected+cropped+resized", "20231126_uop_ycb_stacked")
                save_name_dir = save_name + "stacked/"
                
                os.makedirs(save_name_dir , exist_ok=True)
                hstacked_image.save(save_name_dir + f"stacked_{idx}.png")

                # = close images
                imgs.close()
                hstacked_image.close()

            except:
                pass
                obj_name = one_img_folder_split[-4]
                tag_name = one_img_folder_split[-2]
                print(f"!!>> {obj_name} has not maded")
                # list_not_maded.append(list_not_maded)

    print(list_not_maded)



