import os
import os.path
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
cv2.ocl.setUseOpenCL(False)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

scenes_dicitionary = [ 'chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 19:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            scenes_dir1 = scenes_dicitionary[int(line_split[2])]
            scenes_dir2 = scenes_dicitionary[int(line_split[3])]
            image_name1 = os.path.join(data_root, scenes_dir1, line_split[0][1:])
            image_name2 = os.path.join(data_root, scenes_dir2, line_split[1][1:])
            translation = [0,0,0]
            quaternions = [0,0,0,0]
        else:
            if len(line_split) != 10:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            scenes_dir = scenes_dicitionary[int(line_split[2])]
            image_name1 = os.path.join(data_root, scenes_dir, line_split[0][1:])
            image_name2 = os.path.join(data_root, scenes_dir, line_split[1][1:])
            translation = [float(item) for item in line_split[3:6]]
            quaternions = [float(item) for item in line_split[6:10]]
            
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name1, image_name2, translation, quaternions)
        image_label_list.append(item)
        #print(image_name1, image_name2, translation, quaternions)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SegData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_name1, image_name2, translation, quaternions = self.data_list[index]
        #print(image_name1, image_name2, translation, quaternions)
        image1 = cv2.imread(image_name1, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image1 = np.float32(image1)
        image2 = cv2.imread(image_name2, cv2.IMREAD_COLOR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image2 = np.float32(image2)
        if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_name1 + " " + image_name2 + "\n"))
        if self.transform is not None:
            image1, image2 = self.transform(image1, image2)
        flip = random.uniform(0.0, 1.0)
        if self.split == 'test':
            flip = 1
        if flip > 0.5:
            translation[0] = -translation[0]
            translation[1] = -translation[1]
            translation[2] = -translation[2]
            quaternions[0] = quaternions[0]
            quaternions[1] = -quaternions[1]
            quaternions[2] = -quaternions[2]
            quaternions[3] = -quaternions[3]
            #translation = [float(item) for item in translation]
            #quaternions = [float(item) for indexitem in quaternions]
            image1, image2 = image2, image1
        image1 = np.float32(image1)
        image2 = np.float32(image2)
        return image1, image2, np.array(translation), np.array(quaternions)
