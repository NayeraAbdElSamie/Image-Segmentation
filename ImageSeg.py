#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from zipfile import ZipFile
from os import listdir
from os.path import isfile, join
import scipy.io
from matplotlib import pyplot as plt 
from PIL import Image
from sklearn import preprocessing
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics.cluster import v_measure_score
from pyitlib import discrete_random_variable as drv
import pandas as pd
from PIL import Image,ImageOps
from sklearn.cluster import SpectralClustering


# In[2]:


# unzipping
zip=ZipFile('BSR.zip')
zip.extractall()
zip.close()


# In[3]:


# function reads matlab file and displays the groundtruth segmentations
segmentations=[]
def read_GroundTruth(img_path,gt_path):
    
        mat = scipy.io.loadmat(gt_path,appendmat=False)                     #load matlab file
        groundTruth = mat.get('groundTruth')                                #get image
        label_num = groundTruth.size
        
        #converting the matlab arrays to images
        fig=plt.figure(figsize=(10,10))
        for i in range(label_num):
            trueBoundary = []
            boundary = groundTruth[0][i]['Segmentation'][0][0]
            height = boundary.shape[0]
            width = boundary.shape[1]
            trueBoundary = boundary.reshape(height, width, 1)
            segmentations.append(trueBoundary)
            fig.add_subplot(5,5, i+1)
            plt.imshow(trueBoundary[:,:,0])
        return trueBoundary


# In[4]:


#function that convert images to its RBG representation and apply Kmeans for clustering 
K_segmentations=[]
def KMEANS(img):
    
#convert image to 2D with M*3 dimensions
        x, y, z = img.shape
        image_2d = img.reshape(x*y, z)
        image_2d.shape
        
#Apply K-means on the 2D image
        kmeans_cluster = cluster.KMeans(n_clusters=5)
        kmeans_cluster.fit(image_2d)
        cluster_centers = kmeans_cluster.cluster_centers_
        cluster_labels = kmeans_cluster.labels_
        print(cluster_labels)
        K_segmentations.append(cluster_labels)
        
#Convert back into uint8 and make an image
        centers=np.array(cluster_centers)
        labels=np.transpose(np.uint8(cluster_labels))
        res=centers[labels]
        res=res.reshape(x,y,z)
        fig2 = plt.figure(3)
        plt.imshow(res[:,:,0])
        plt.show()
        return cluster_labels


# In[5]:


# F_Measure

# Handling dividing by NaN
np.seterr(divide='ignore', invalid='ignore')

def f_measure_score(ground_truth, pred):
    
    #Building Confusion Matrix
    cm = metrics.confusion_matrix(ground_truth, pred)
    
    # Computing F_measure
    TP = np.diag(cm)                     # True Positives are on the diagonal position
    FP = np.sum(cm, axis=0) - TP         # False Positives are column-wise sums. Without the diagonal
    FN = np.sum(cm, axis=1) - TP         # False Negatives are row-wise sums. Without the diagonal
    
    P = (TP/(TP+FP))                 # Precision
    R = (TP/(TP+FN))                 # Recall
    F = (2*P*R) / (P+R)
    
    #print (metrics.accuracy_score(ground_truth, pred))
    #print ('Report : ')
    #print (metrics.classification_report(ground_truth, pred)) 
    F = pd.DataFrame({'col1': F})
    print(sum(F['col1'].dropna().unique()))


# In[6]:


# Conditional Entropy
def cond_ent_axis(row):
    sum_of_row = sum(row)
    cond_ent_of_row = 0
    for i in range(len(row)):
        cond_ent_of_row += (row[i] / sum_of_row) * np.log(row[i] / sum_of_row)
    result = sum_of_row * cond_ent_of_row
    return result

def conditional_entropy(ground_truth, pred):
    
    pred_labels_size = pred_labels.size
    
    # Building Contingency Matrix
    con_matrix = metrics.cluster.contingency_matrix(boundary, pred_labels)
    # Rows represent predictions
    # Columns represent ground truth
    
    cond_ent = pd.DataFrame({'col1': np.apply_along_axis(cond_ent_axis, axis=1, arr=con_matrix)})
    #print(np.apply_along_axis(cond_ent_axis, axis=1, arr=con_matrix))
    #print(sum(np.apply_along_axis(cond_ent_axis, axis=1, arr=con_matrix)))
    
    cond_ent = cond_ent / pred_labels_size
    
    # Ignoring nan
    #cond_ent['col1'].dropna().unique()
    print(-(sum(cond_ent['col1'].dropna().ffill())))


# In[7]:


#function to evaluate segmentations using F-measures and conditional Entropy
def evaluate(boundary, pred_labels):
    #print(metrics.f1_score(boundary, pred_labels, average='macro'))
    print("Conditional Entropy:")
    print(conditional_entropy(boundary, pred_labels))
    print("F-Measure:")
    f_measure_score(boundary, pred_labels)
    #drv.entropy_conditional(boundary, pred_labels)


# In[8]:


#normalized cut ussing 5-NN graph
def Normalized_cut(img):
    
        img=np.array(img)
        x, y, z = img.shape
        image_2d = img.reshape(x*y, z)
        image_2d.shape
        Norm_cut=SpectralClustering(n_clusters=7, eigen_solver=None, random_state=0,affinity='nearest_neighbors', n_neighbors=7).fit(image_2d)
        Norm_labels=Norm_cut.labels_
        Norm_affinity=Norm_cut.affinity_matrix_

        #Convert back into uint8 and make an image
        aff=np.array(Norm_affinity)
        labels=np.transpose(np.uint8(Norm_labels))
        res=labels.reshape(x,y,1)
        Res=res.reshape(x,y,1)
        fig2 = plt.figure(3)
        plt.imshow(res[:,:,0])
        plt.show()


# In[9]:


#directories
train_dir = "D:/College/8th term/Pattern Recognition/Assignments/Assignment 2/BSR/BSDS500/data/images/train/"
train_gt_dir = "D:/College/8th term/Pattern Recognition/Assignments/Assignment 2/BSR/BSDS500/data/groundTruth/train/"
test_dir = "D:/College/8th term/Pattern Recognition/Assignments/Assignment 2/BSR/BSDS500/data/images/test/"
test_gt_dir = "D:/College/8th term/Pattern Recognition/Assignments/Assignment 2/BSR/BSDS500/data/groundTruth/test/"

#generating images ang gt images directories
img_files = [f for f in listdir(train_dir) if isfile(join(train_dir, f)) and f.endswith(".jpg")]
gt_files = [f for f in listdir(train_gt_dir) if isfile(join(train_gt_dir, f)) and f.endswith(".mat")]

#global variables
segmentations=[]
K_segmentations=[]

#reading images and groundTruth

for i in range(5):
        print("Handling:   " + gt_files[i] + "   ====   " + img_files[i])
        img_path = train_dir + img_files[i]
        gt_path = train_gt_dir + gt_files[i]
        img=plt.imread(img_path)                                          #reading image from img-path
        fig1=plt.figure(1)
        plt.imshow(img)
        plt.show()
        trueBoundary = read_GroundTruth(img_path,gt_path)     #to display  ground truth
        print(trueBoundary.shape)
        x, y, z = trueBoundary.shape
        pred_labels=KMEANS(img)                           #for clustering
        im = Image.open(img_path)
        size=(100,100)
        image=ImageOps.fit(im, size, Image.ANTIALIAS)
        plt.imshow(image)
        plt.show()
        
        print("Normalized cut:")
        Normalized_cut(image)
        print(pred_labels.shape)
        boundary =trueBoundary.reshape(x*y,z)
        boundary=boundary.flatten()
        
        print('Scores')
        evaluate(boundary, pred_labels)
        print('\n')


# In[ ]:




