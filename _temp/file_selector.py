

# private


import os
import glob
import shutil 

import tqdm
import natsort


if __name__=="__main__":

    data_root = "/home/ailab-ur5/_Workspaces/_data/uop/20231124_uop_ycb_rotate/ycb/"
    dst_root = "/home/ailab-ur5/_Workspaces/_data/uop/20231125_uop_ycb_rotate_selected/ycb/"


    # list_obj_folder = os.listdir(data_root)
    list_obj_folder = glob.glob(data_root + "*/")
    sorted_list_obj_folder = natsort.natsorted(list_obj_folder)
    
    list_not_copied = []
    for one_obj_folder in tqdm.tqdm(sorted_list_obj_folder):
        list_img_folder = glob.glob(one_obj_folder + "recorded_data/*/")
        sorted_list_img_folder = natsort.natsorted(list_img_folder)

        for one_img_folder in sorted_list_img_folder:
            
            list_img_file = glob.glob(one_img_folder + "*.png")
            sorted_list_img_file = natsort.natsorted(list_img_file)

            for one_img in sorted_list_img_file:
                img_num = one_img.split("/")[-1].replace(".png", "")
                if 419 <= int(img_num):
                    continue

                try:
                    saved_dir = one_img.replace(data_root, dst_root)
                    os.makedirs(saved_dir.replace(saved_dir.split("/")[-1], ""), exist_ok=True)
                    shutil.copyfile(one_img, saved_dir)
                    # shutil.copyfile(one_data, saved_dir + f"/label_inspected{data_dir[-1]}")
                except:
                    print(f"!!>> {one_img} has not copied")
                    list_not_copied.append(list_not_copied)
    
    print(list_not_copied)



            
        








