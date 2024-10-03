#!/usr/bin/env python

import sys
import os
import os.path as osp
import shutil
from collections import defaultdict
import random 


def split_data(file_list, train_ratio, val_ratio):
    random.shuffle(file_list)  # Shuffle the file list
    total = len(file_list)
    
    train_split = int(total * train_ratio)
    val_split = train_split + int(total * val_ratio)
    
    train_files = file_list[:train_split]
    val_files = file_list[train_split:val_split]  # Remaining files for validation
    test_files = file_list[train_split:]  # Remaining files for validation
    
    return train_files, val_files, test_files 

def copyfile(srcfile, dstfile):
    if not osp.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath, fname = osp.split(dstfile)
        if not osp.exists(fpath):
            os.makedirs(fpath)
        shutil.copyfile(srcfile, dstfile)

def movetodir(record_file, Data_root, dataset_path):
    with open(record_file, 'r') as f:
        for line in f:
            src_img_path = osp.join(Data_root, line.rstrip())
            dst_img_path = osp.join(dataset_path, line.rstrip())
            copyfile(src_img_path, dst_img_path)

def main():
    Data_root = "/workspace/XAI606-project/dataset/png"
    Train_record_file = "/workspace/XAI606-project/dataset/png/train.txt"
    Val_record_file = "/workspace/XAI606-project/dataset/png/val.txt"
    Test_record_file = "/workspace/XAI606-project/dataset/png/test.txt"

    class_files = defaultdict(list)
    with open(os.path.join(Data_root, 'filelist.txt'), 'r') as file:
        for line in file:
            line = line.strip()
            class_name = line.split('/')[0]
            class_files[class_name].append(line)

    train_ratio = 0.7  # 70% for training
    val_ratio = 0.15    # 15% for validation
    test_ratio = 0.15 

    # Containers for final splits
    train_data = []
    val_data = []
    test_data = []

    Dataset_train_path = osp.join(Data_root, "../train_val/train")
    if not osp.exists(Dataset_train_path):
        os.makedirs(Dataset_train_path)
    
    Dataset_val_path = osp.join(Data_root, "../train_val/val")
    if not osp.exists(Dataset_val_path):
        os.makedirs(Dataset_val_path)

    Dataset_test_path = osp.join(Data_root, "../train_val/test")
    if not osp.exists(Dataset_test_path):
        os.makedirs(Dataset_test_path)

    for class_name, files in class_files.items():
        train_files, val_files, test_files = split_data(files, train_ratio, val_ratio)
        train_data.extend(train_files)
        val_data.extend(val_files)
        test_data.extend(test_files)

    # Write the splits to separate files
    with open(Train_record_file, 'w') as file:
        for item in train_data:
            file.write(f"{item}\n")

    with open(Val_record_file, 'w') as file:
        for item in val_data:
            file.write(f"{item}\n")
            
    with open(Test_record_file, 'w') as file:
        for item in val_data:
            file.write(f"{item}\n")

    movetodir(Train_record_file, Data_root, Dataset_train_path)
    movetodir(Val_record_file, Data_root, Dataset_val_path)
    movetodir(Test_record_file, Data_root, Dataset_test_path)

    print(f"Training data: {len(train_data)} files written to {Train_record_file}")
    print(f"Validation data: {len(val_data)} files written to {Val_record_file}")
    print(f"Validation data: {len(val_data)} files written to {Test_record_file}")



if __name__ == "__main__":
    main()