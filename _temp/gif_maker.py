
# private
# pip install opencv-python-headless natsort

import os
import glob
import shutil 

# import cv2
from PIL import Image
import tqdm
import natsort

    
if __name__=="__main__":

    data_root = "/home/ailab-ur5/_Workspaces/_data/uop/20231125_uop_ycb_rotate_selected/ycb/"
    # dst_root = "/home/ailab-ur5/_Workspaces/_data/uop/20231125_uop_ycb_rotate_selected/ycb_gif/"


    # list_obj_folder = os.listdir(data_root)
    list_obj_folder = glob.glob(data_root + "*/")
    sorted_list_obj_folder = natsort.natsorted(list_obj_folder)
    
    list_not_maded = []
    for idx, one_obj_folder in tqdm.tqdm(enumerate(sorted_list_obj_folder)):
        
        list_img_folder = glob.glob(one_obj_folder + "recorded_data/*/")
        sorted_list_img_folder = natsort.natsorted(list_img_folder)

        for one_img_folder in sorted_list_img_folder:
            
            list_img_file = glob.glob(one_img_folder + "*.png")
            sorted_list_img_file = natsort.natsorted(list_img_file)

            one_img_folder_split = one_img_folder.split("/")

            try:
                pil_images = [Image.open(x) for x in sorted_list_img_file]
                pil_images_first = pil_images[0].copy()

                gif_tag = one_img_folder_split[-2]
                pil_images_first.save(one_obj_folder + f"{gif_tag}.gif", save_all=True, append_images=pil_images[1:], loop=0xff, duration=10)
            except:
                obj_name = one_img_folder_split[-4]
                tag_name = one_img_folder_split[-2]
                print(f"!!>> {obj_name}_{tag_name} has not maded")
                list_not_maded.append(list_not_maded)

    
    print(list_not_maded)