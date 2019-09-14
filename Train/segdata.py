import os
import os.path
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset

cv2.ocl.setUseOpenCL(False)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

scenes_dicitionary = [ 'chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']

def sgn(x):
        #signum function
        if(x==0):
                y = 0
        elif (x>0):
                y = 1
        else:
                y = -1

        return y


def rotm2quat(m_A): #returns a quaternion whose scalar part is positive to keep angle between -180 to +180 deg.
                                        #formula from pg 97 of textbook by Junkins
        q0 = 1 + np.trace(m_A)
        q1 = 1 + m_A[0,0] - m_A[1,1] - m_A[2,2]
        q2 = 1 - m_A[0,0] + m_A[1,1] - m_A[2,2]
        q3 = 1 - m_A[0,0] - m_A[1,1] + m_A[2,2]
        qm = max(m_A[0,0],m_A[1,1],m_A[2,2])
        if(q0>0):
                q0 = math.sqrt(q0)/2
                q1 = -(m_A[1,2] - m_A[2,1])/(4*q0)
                q2 = -(m_A[2,0] - m_A[0,2])/(4*q0)
                q3 = -(m_A[0,1] - m_A[1,0])/(4*q0)

        elif(qm==m_A[0,0]):
                q1 = math.sqrt(q1)/4
                q0 = -(m_A[1,2] - m_A[2,1])/(4*q1)
                q2 = (m_A[0,2] + m_A[2,0])/(4*q1)
                q3 = (m_A[0,1] + m_A[1,0])/(4*q1)

        elif(qm==q2):
                q2 = math.sqrt(q2)/4
                q0 = -(m_A[2,0] - m_A[0,2])/(4*q2)
                q1 = (m_A[0,1] + m_A[1,0])/(4*q2)
                q3 = (m_A[1,2] + m_A[2,1])/(4*q2)

        else:
                q3 = math.sqrt(q3)/4
                q0 = -(m_A[0,1] - m_A[1,0])/(4*q3)
                q1 = (m_A[0,2] + m_A[2,0])/(4*q3)
                q2 = (m_A[1,2] - m_A[2,1])/(4*q3)

        v_q = np.array([q0,q1,q2,q3])
        v_q = v_q/np.linalg.norm(v_q)
        if (sgn(v_q[0])==-1):
                v_q = -1 * v_q
        return v_q

    
def quaternion_r(a1, a2):
    a1 = np.mat(a1)
    a2 = np.mat(a2)
    return rotm2quat(a1.I.dot(a2))
def quaternion_t(t1, t2):
    return t2-t1
def quaternion(r1,r2,t1,t2):
    return quaternion_t(t1,t2), quaternion_r(r1,r2)


def get_pose(filename):
    #f = open(filename).read()
    matrix =np.loadtxt(filename, delimiter="\t ")
    r = matrix[:3,:3]
    t = matrix[:3, 3]
    return r, t, matrix






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
            if len(line_split) != 13:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            anchor_name = os.path.join(data_root, line_split[0])
            image1_name = os.path.join(data_root, line_split[1])
            image1_r = [float(item) for item in line_split[2:5]]
            image1_r[2] /= 180.0
            image2_name = os.path.join(data_root, line_split[5])
            image2_r = [float(item) for item in line_split[6:9]]
            image2_r[2] /= 180.0
            image3_name = os.path.join(data_root, line_split[9])
            image3_r = [float(item) for item in line_split[10:13]]
            image3_r[2] /= 180.0
            r_anchor, t_anchor, _ = get_pose(anchor_name)
            r_1, t_1, _ = get_pose(image1_name)
            r_2, t_2, _ = get_pose(image2_name)
            r_3, t_3, _ = get_pose(image3_name)
            relative_t1, relative_r1 = quaternion(r_anchor,r_1,t_anchor,t_1)
            relative_t2, relative_r2 = quaternion(r_anchor,r_2,t_anchor,t_2)
            relative_t3, relative_r3 = quaternion(r_anchor,r_3,t_anchor,t_3)
            absolute_r1 = rotm2quat(np.mat(r_1))
            absolute_r2 = rotm2quat(np.mat(r_2))
            absolute_r3 = rotm2quat(np.mat(r_3))
            absolute_ranchor = rotm2quat(np.mat(r_anchor))
            
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        
        item = (anchor_name.replace('pose.txt','color.png'), image1_name.replace('pose.txt','color.png'), image2_name.replace('pose.txt','color.png'), image3_name.replace('pose.txt','color.png'), relative_t1, relative_r1, relative_t2, relative_r2, relative_t3, relative_r3, image1_r, image2_r, image3_r, absolute_r1, t_1, absolute_r2, t_2, absolute_r3, t_3, absolute_ranchor, t_anchor)
        image_label_list.append(item)
#         print(item)
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
        anchor_name, image1_name, image2_name, image3_name, relative_t1, relative_r1, relative_t2, relative_r2, relative_t3, relative_r3, image1_r, image2_r, image3_r, absolute_r1, absolute_t1, absolute_r2, absolute_t2, absolute_r3, absolute_t3, absolute_ranchor, absolute_tanchor = self.data_list[index]
        image_anchor = cv2.imread(anchor_name, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image_anchor = cv2.cvtColor(image_anchor, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image_anchor = np.float32(image_anchor)
        image1 = cv2.imread(image1_name, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image1 = np.float32(image1)
        image2 = cv2.imread(image2_name, cv2.IMREAD_COLOR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = np.float32(image2)
        image3 = cv2.imread(image3_name, cv2.IMREAD_COLOR)
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
        image3 = np.float32(image3)
        if image_anchor.shape != image2.shape or image_anchor.shape != image2.shape or image_anchor.shape != image3.shape:
            raise (RuntimeError("Image & label shape mismatch: " + image_name1 + " " + image_name2 + "\n"))
        if self.transform is not None:
            image_anchor, image1, image2, image3 = self.transform(image_anchor, image1, image2, image3)
            
        image1_r, image2_r, image3_r = np.array(image1_r), np.array(image2_r), np.array(image3_r)
        absolute_r1, absolute_t1, absolute_r2, absolute_t2, absolute_r3, absolute_t3, absolute_ranchor, absolute_tanchor = np.array(absolute_r1).astype(np.float32), np.array(absolute_t1).astype(np.float32), np.array(absolute_r2).astype(np.float32), np.array(absolute_t2).astype(np.float32), np.array(absolute_r3).astype(np.float32), np.array(absolute_t3).astype(np.float32), np.array(absolute_ranchor).astype(np.float32), np.array(absolute_tanchor).astype(np.float32)
#         flip = random.uniform(0.0, 1.0)
#         if self.split == 'test':
#             flip = 1
#         if flip > 0.5:
#             translation[0] = -translation[0]
#             translation[1] = -translation[1]
#             translation[2] = -translation[2]
#             quaternions[0] = quaternions[0]
#             quaternions[1] = -quaternions[1]
#             quaternions[2] = -quaternions[2]
#             quaternions[3] = -quaternions[3]
#             #translation = [float(item) for item in translation]
#             #quaternions = [float(item) for indexitem in quaternions]
#             image1, image2 = image2, image1
        return image_anchor, image1, image2, image3, relative_t1, relative_r1, relative_t2, relative_r2, relative_t3, relative_r3, image1_r, image2_r, image3_r, anchor_name.replace('color.png', 'pose.txt'), absolute_r1, absolute_t1, absolute_r2, absolute_t2, absolute_r3, absolute_t3, absolute_ranchor, absolute_tanchor
