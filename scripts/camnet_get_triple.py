import matplotlib.pyplot as plt
import cv2
import numpy as np
def get_pose(filename):
    #f = open(filename).read()
    matrix =np.loadtxt(filename, delimiter="\t ")
    r = matrix[:3,:3]
    t = matrix[:3, 3]
    return r, t, matrix



#https://blog.csdn.net/lql0716/article/details/72597719
import math
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

def quat2rotm(v_q): #given a quaternion it returns a rotation matrix
	q1 = v_q[1]  
	q2 = v_q[2]
	q3 = v_q[3]
	q0 = v_q[0]

	m_M1 = 2* np.array([[-q2**2 - q3**2,q1*q2,q1*q3],[q1*q2,-q1**2 - q3**2,q2*q3],[q1*q3,q2*q3,-q1**2-q2**2]])
	m_M2 = 2*q0*np.array([[0,-q3,q2],[q3,0,-q1],[-q2,q1,0]])
	m_M3 = np.identity(3)  #norm(v_q)=1 
	return m_M1 + m_M2 + m_M3


def relative_r(a1, a2):
    a1 = np.mat(a1)
    a2 = np.mat(a2)
    return np.arccos((np.trace(a1.I.dot(a2))-1)/2)*180/np.pi
def relative_t(t1, t2):
    return np.linalg.norm(t2-t1) 
def quaternion_r(a1, a2):
    a1 = np.mat(a1)
    a2 = np.mat(a2)
    return rotm2quat(a1.I.dot(a2))
def quaternion_t(t1, t2):
    return t2-t1
def quaternion(r1,r2,t1,t2):
    return np.concatenate((quaternion_t(t1,t2),quaternion_r(r1,r2)))

def reprojection(depth, r1, r2, t1, t2, h=480, w=640):
    K = np.mat([[585,0,320],[0,585,240],[0,0,1]])
    x_list = []
    depth_list = []
    for i in range(0,h,10):
        for j in range(0,w,10):
            if depth[i][j]!=0 and depth[i][j]!=65535:
                x_list.append([j,i,1])
                depth_list.append(depth[i][j])
    x1 = np.mat(x_list)
    dep = np.mat(depth_list)
    x_new = np.multiply((K.I.dot(x1.T)),(dep/1000))
    x_new1 = np.mat(r2).I.dot(np.mat(r1).dot(x_new)+ (np.mat(t1)-np.mat(t2)).T)
    result = np.array(K.dot(x_new1))
    x2 = result/result[2]
    flag = 0
    for i in range(len(x_list)):
        if (x2[0][i] > 0) and (x2[0][i] < w) and (x2[1][i] > 0) and (x2[1][i] < h):
            flag += 1
    #print(x_list[0],x2[:,0])
    return float(flag)/len(x_list)


import os
import random
import sys
scenes_dicitionary = [ 'chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']   #stair 500
scenes_train_set = []
test_set = []
all_train_set = []
for scene in scenes_dicitionary:
    train_split = [item.strip()[8:] for item in open(scene + '/TrainSplit.txt').readlines()]
    test_split = [item.strip()[8:] for item in open(scene + '/TestSplit.txt').readlines()]
    for index in test_split:
        test_set.extend([scene + '/seq-%02d/' % int(index) + item for item in os.listdir(scene + '/seq-%02d' % int(index)) if 'pose' in item])
    train_set = []
    for index in train_split:
        train_set.extend([scene + '/seq-%02d/' % int(index) + item for item in os.listdir(scene + '/seq-%02d' % int(index)) if 'pose' in item])
        all_train_set.extend([scene + '/seq-%02d/' % int(index) + item for item in os.listdir(scene + '/seq-%02d' % int(index)) if 'pose' in item])
    scenes_train_set.append(train_set)


for item in all_train_set[int(sys.argv[1]):int(sys.argv[2])]:
    scene_index = scenes_dicitionary.index(item.split('/')[0])
    r1, t1, all1 = get_pose(item)
    depth = cv2.imread(item.replace('pose.txt','depth.png'),2)
    train_set = scenes_train_set[scene_index]
    set1 = []
    set2 = []
    set3 = []
    for index in range(random.randint(0,int(len(train_set)/100)), len(train_set),int(len(train_set)/100)):
        r2, t2, all2 = get_pose(train_set[index])
        rate1 = reprojection(depth, r1, r2, t1, t2)
        depth = cv2.imread(train_set[index].replace('pose.txt','depth.png'),2)
        rate2 = reprojection(depth, r2, r1, t2, t1)
        rater = relative_r(r1,r2)
        if rate1> 0.4 and rate2>0.4 and rater< 30:
            set1.append([train_set[index], rate1, rate2, relative_r(r1,r2)])
        elif rate1> 0.3 and rate2>0.3 and rater> 60:
            set2.append([train_set[index], rate1, rate2, relative_r(r1,r2)])
        elif rate1< 0.25 and rate2<0.25 and rate1 > 0.05 and rate2 > 0.05: 
            set3.append([train_set[index], rate1, rate2, relative_r(r1,r2)])
    if set1 and set2 and set3:
        print(item, end=' ')
        for parameter in set1[random.randint(0,len(set1)-1)]:
            print(parameter, end=' ')
        for parameter in set2[random.randint(0,len(set2)-1)]:
            print(parameter, end=' ')
        for parameter in set3[random.randint(0,len(set3)-1)]:
            print(parameter, end=' ')
        print ()
    
        write_dir = 'retrival_lists/' + item[:-21]
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        write_file = open('retrival_lists/' + item, 'w')
        for reference_item in set1:
            r_refer, t_refer, all_refer = get_pose(reference_item[0])
            print(reference_item[0], file=write_file, end = ' ')
            for r_item in rotm2quat(np.mat(r_refer)):
                print(r_item, file=write_file, end = ' ')
            for t_item in t_refer:
                print(t_item, file=write_file, end = ' ')
            print (file=write_file)
        for reference_item in set2:
            r_refer, t_refer, all_refer = get_pose(reference_item[0])
            print(reference_item[0], file=write_file, end = ' ')
            for r_item in rotm2quat(np.mat(r_refer)):
                print(r_item, file=write_file, end = ' ')
            for t_item in t_refer:
                print(t_item, file=write_file, end = ' ')
            print (file=write_file)
        for reference_item in set3:
            r_refer, t_refer, all_refer = get_pose(reference_item[0])
            print(reference_item[0], file=write_file, end = ' ')
            for r_item in rotm2quat(np.mat(r_refer)):
                print(r_item, file=write_file, end = ' ')
            for t_item in t_refer:
                print(t_item, file=write_file, end = ' ')
            print (file=write_file)
        write_file.close()