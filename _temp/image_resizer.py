
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

    
if __name__=="__main__":

    data_root = "/home/ailab-ur5/_Workspaces/_data/uop/20231125_uop_ycb_rotate_selected/ycb/"
    dst_root_1 = "/home/ailab-ur5/_Workspaces/_data/uop/20231126_uop_ycb_selected+cropped/ycb/"
    dst_root_2 = "/home/ailab-ur5/_Workspaces/_data/uop/20231126_uop_ycb_selected+cropped+resized/ycb/"


    # list_obj_folder = os.listdir(data_root)
    list_obj_folder = glob.glob(data_root + "*/")
    sorted_list_obj_folder = natsort.natsorted(list_obj_folder)
    
    list_not_maded = []
    for idx, one_obj_folder in tqdm.tqdm(enumerate(sorted_list_obj_folder)):

        if idx < 23 :
            continue
        
        list_img_folder = glob.glob(one_obj_folder + "recorded_data/*/")
        sorted_list_img_folder = natsort.natsorted(list_img_folder)

        for one_img_folder in sorted_list_img_folder:
            
            list_img_file = glob.glob(one_img_folder + "*.png")
            sorted_list_img_file = natsort.natsorted(list_img_file)

            one_img_folder_split = one_img_folder.split("/")

            # # = for visualize
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # plt.ion()
            # plt.show()

            for one_image_file in tqdm.tqdm(sorted_list_img_file):
                try:
                    pil_image = Image.open(one_image_file).copy()

                    # = crop
                    x_low = int((1920-1080) / 2)
                    y_low = 0
                    x_high = 1080 + int((1920-1080) / 2)
                    y_high = 1080
                    crop_box = (x_low, y_low, x_high, y_high)
                    crop_pil_image = pil_image.crop(crop_box).copy()
                    # # = show
                    # plt_img = ax.imshow(crop_pil_image)   # : plt
                    # plt.draw()
                    # plt.imshow(crop_pil_image)
                    # crop_pil_image.show()                 # : PIL
                    save_name_1 = one_image_file.replace("20231125_uop_ycb_rotate_selected", "20231126_uop_ycb_selected+cropped")
                    save_name_1_dir = save_name_1.replace(save_name_1.split("/")[-1], "")
                    
                    os.makedirs(save_name_1_dir, exist_ok=True)
                    crop_pil_image.save(save_name_1)

                    # = resize
                    resize_wh = (512, 512)
                    resized_pil_image = crop_pil_image.resize(resize_wh, resample=1).copy() # : nn resize
                    # # = show
                    # plt_img = ax.imshow(resized_pil_image)   # : plt
                    # plt.draw()
                    # plt.imshow(crop_pil_image)                # : or
                    # resized_pil_image.show()                  # : PIL
                    
                    save_name_2 = one_image_file.replace("20231125_uop_ycb_rotate_selected", "20231126_uop_ycb_selected+cropped+resized")
                    save_name_2_dir = save_name_2.replace(save_name_1.split("/")[-1], "")
                    
                    os.makedirs(save_name_2_dir, exist_ok=True)
                    resized_pil_image.save(save_name_2)

                    # = add close for process
                    pil_image.close()
                    crop_pil_image.close()
                    resized_pil_image.close()

                except:
                    pass
                    obj_name = one_img_folder_split[-4]
                    tag_name = one_img_folder_split[-2]
                    print(f"!!>> {obj_name}_{tag_name} has not maded")
                    # list_not_maded.append(list_not_maded)

    
    print(list_not_maded)