

# private


import os
import glob
import shutil 

import tqdm
import natsort


if __name__=="__main__":

    data_root = "F:\_device\_local\20231124_uop_ycb_rotate"
    dst_root = "/home/ailab-ur5/_Workspaces/_data/uop/pub_dataset"

    list_file_folders = glob.glob(data_root+'*[!.json]')

    sorted_list_file_folders = natsort.natsorted(list_file_folders)

    list_not_copied = [] 
    for one_dataset_folder in sorted_list_file_folders:

        datas_folder = glob.glob(one_dataset_folder + '/*[!.png]')

        print ()

        sorted_datas_folder = natsort.natsorted(datas_folder)
        for one_data_folder in tqdm.tqdm(sorted_datas_folder) :

            datas = glob.glob(one_data_folder + '/[label]*')
            # datas = glob.glob(one_data_folder + '/[label_inspected]*')    # : dosen't work
            for one_data in datas:
                # if "label_inspected" in one_data:
                if "label_inspected" not in one_data:
                    continue
                try:
                    data_dir = one_data.split("/label")
                    # data_dir = one_data.split("/label_inspected")
                    saved_dir = data_dir[0].replace(
                        "/media/ailab-ur5/T7/_storage/Server/NAS/ailab(PAT)/Dataset/SOP/dataset", 
                        dst_root)
                    os.makedirs(saved_dir, exist_ok=True)
                    shutil.copyfile(one_data, saved_dir + f"/label{data_dir[-1]}")
                    # shutil.copyfile(one_data, saved_dir + f"/label_inspected{data_dir[-1]}")
                except:
                    print(f"!!>> {one_data} has not copied")
                    list_not_copied.append(one_data)
            # print ()
        # print ()

    print(list_not_copied)

    print()






