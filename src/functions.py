"""
The file contains functions that will be used in the ipynb files. This file is only used to add the notes
and generate the documentation. These functions have been added to the ipynb files

@author: Fan Yang
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import space_filling_decomp_new as sfc
import cv2
import sys
import vtk
import vtktools
import numpy as np
import pandas as pd
import time
import glob
import progressbar
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.tri as tri
import meshio
import re
import math

# create an animation
from matplotlib import animation
from IPython.display import HTML
# custom colormap
import cmocean

import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, TensorDataset, Dataset


#################################### Functions for generate the SFC ordering, Written by YuJin ####################################
def read_in_files(data_path, file_format='vtu', vtu_fields=None):
    '''
    This function reads in the vtu/txt files in a {data_path} as tensors, of shape [snapshots, number of Nodes, Channels]
    This function is written by YuJin

    Parameters
    ---
    data_path: [string] the data_path which holds vtu/txt files, no other type of files are accepted!!!
    file_format: [string] 'vtu' or 'txt', the format of the file.
    vtu_fields: [list] the list of vtu_fields if read in vtu files, the last dimension of the tensor, e.g. ['Velocity', 'Pressure']

    Returns
    ---
    Case 1 - file_format='vtu': (3-tuple) [torch.FloatTensor] full_stage over times step, time along 0 axis; [torch.FloatTensor] coords of the mesh; [dictionary] cell_dict of the mesh.

    Case 2 - file_format='txt': [torch.FloatTensor] full_stage over times step, time along 0 axis

    '''
    data = glob.glob(data_path + "*")
    num_data = len(data)
    file_prefix = data[0].split('.')[-2].split('_')
    file_prefix.pop(-1)
    if len(file_prefix) != 1: file_prefix = '_'.join(file_prefix) + "_"
    else: file_prefix = file_prefix[0] + "_"
    file_format = '.' + file_format
    print('file_prefix: %s, file_format: %s' % (file_prefix, file_format))
    cnt_progress = 0
    if (file_format == ".vtu"):
        print("Read in vtu Data......\n")
        bar=progressbar.ProgressBar(maxval=num_data)
        bar.start()
        data = []
        coords = None
        cells = None
        start = 0
        while(True):
            if not os.path.exists(F'{file_prefix}%d{file_format}' % start):
                print(F'{file_prefix}%d{file_format} not exist, starting number switch to {file_prefix}%d{file_format}' % (start, start+1))
                start += 1
            else: break
        for i in range(start, num_data + start):
            data.append([])
            vtu_file = meshio.read(F'{file_prefix}%d{file_format}' % i)
            if not (coords == vtu_file.points).all():
               coords = vtu_file.points
               cells = vtu_file.cells_dict
               print('mesh adapted at snapshot %d' % i)
            for j in range(len(vtu_fields)):
                vtu_field = vtu_fields[j]
                if not vtu_field in vtu_file.point_data.keys():
                   raise ValueError(F'{vtu_field} not avaliable in {vtu_file.point_data.keys()} for {file_prefix} %d {file_format}' % i)
                field = vtu_file.point_data[vtu_field]
                if j == 0:
                   if field.ndim == 1: field = field.reshape(field.shape[0], 1)
                   data[i - start] = field
                else:
                   if field.ndim == 1: field = field.reshape(field.shape[0], 1)
                   data[i - start] = np.hstack((data[i - start], field))
            cnt_progress +=1
            bar.update(cnt_progress)
        bar.finish()
        whole_data = torch.from_numpy(np.array(data)).float()
        
        # get rid of zero components
        zero_compos = 0
        for i in range(whole_data.shape[-1]):
            if whole_data[..., i].max() - whole_data[..., i].min() < 1e-8:
               zero_compos += 1
               whole_data[..., i:-1] = whole_data[..., i + 1:]
        if zero_compos > 0 : whole_data = whole_data[..., :-zero_compos]
        
        return whole_data, coords, cells    

    elif (file_format == ".txt" or file_format == ".dat"):
        print("Read in txt/dat Data......")
        bar=progressbar.ProgressBar(maxval=num_data)
        data = []
        for i in range(num_data):
            data[i] = torch.from_numpy(np.loadtxt('{file_prefix} %d {file_format}' % i)).float()
            cnt_progress +=1
            bar.update(cnt_progress)
        bar.finish()
        return torch.cat(data, -1)


def get_sfc_curves_from_coords(coords, num):
    '''
    This functions generate space-filling orderings for a coordinate input of a Discontinuous Galerkin unstructured mesh.
    This function is written by YuJin
    Parameters
    ---
    coords: [2d-array] coordinates of mesh, read from meshio.read().points or vtktools.vtu().GetLocations(),  of shape(number of Nodes, 3)
    num: [int] the number of (orthogonal) space-filling curves you want.

    Returns
    ---
    curve_lists: [list of 1d-arrays] the list of space-filling curves, each element of shape [number of Nodes, ]
    inv_lists: [list of 1d-arrays] the list of inverse space-filling curves, each element of shape [number of Nodes, ]
    '''
    findm, colm, ncolm = sfc.form_spare_matric_from_pts(coords, coords.shape[0])
    colm = colm[:ncolm]
    curve_lists = []
    inv_lists = []
    ncurve = num
    graph_trim = -10  # has always been set at -10
    starting_node = 0 # =0 do not specifiy a starting node, otherwise, specify the starting node
    whichd, space_filling_curve_numbering = sfc.ncurve_python_subdomain_space_filling_curve(colm, findm, starting_node, graph_trim, ncurve, coords.shape[0], ncolm)
    for i in range(space_filling_curve_numbering.shape[-1]):
        curve_lists.append(np.argsort(space_filling_curve_numbering[:,i]))
        inv_lists.append(np.argsort(np.argsort(space_filling_curve_numbering[:,i])))

    return curve_lists, inv_lists


####################################Functions used in Flow past cylinder####################################
def saveIndex(path_train, path_valid, path_test,train_index, valid_index, test_index):
    """
    The indexes of training, valid and test dataset are generated randomly. The indexes 
    are saved as csv file so that these indexed can be reused in loading data. 

    Parameters
    ----------
    path_train : string
        The path of the train_index will be saved
    path_valid : string
        The path of the valid_index will be saved
    path_test : string
        The path of the test_index will be saved
    train_index: array
        The array generated from the index_split function
    valid_index: array
        The array generated from the index_split function
    test_index: array
        The array generated from the index_split function
    """
 
    np.savetxt(path_train,train_index, delimiter=',')
    np.savetxt(path_valid,valid_index, delimiter=',')
    np.savetxt(path_test,test_index, delimiter=',')

def getIndex(path_train,path_valid,path_test):
    """
    Read the indexes of training,valid and test dataset from the csv file 

    Parameters
    ----------
    path_train : string
        The path of the train_index will be loaded
    path_valid : string
        The path of the valid_index will be loaded
    path_test : string
        The path of the test_index will be loaded
   
    Returns
    ----------
    train_index: array
        Load the file to the array
    valid_index: array
        Load the file to the array
    test_index: array
        Load the file to the array
    """
    train_index = np.loadtxt(path_train,delimiter=",")
    valid_index = np.loadtxt(path_valid,delimiter=",")
    test_index = np.loadtxt(path_test,delimiter=",")
    return train_index,valid_index,test_index

def saveMode(path_train, path_valid, path_test,mode_train, mode_valid, mode_test):
    """
    # The output of the encoder is called mode. Save the mode of training data, valid
    # data and test respectively. This function will be used in hierarchical autoencoder.

    Parameters
    ----------
    path_train : string
        The path of the mode_train will be saved
    path_valid : string
        The path of the mode_valid will be saved
    path_test : string
        The path of the mode_test will be saved
    mode_train: tensor
        The tensor generated from the output the the encoder using training data
    mode_valid: tensor
        The tensor generated from the output the the encoder using valid data
    mode_test: tensor
        The tensor generated from the output the the encoder using test data
    """   
    np.savetxt(path_train,mode_train.cpu().data.numpy(), delimiter=',')
    np.savetxt(path_valid,mode_valid.cpu().data.numpy(), delimiter=',')
    np.savetxt(path_test,mode_test.cpu().data.numpy(), delimiter=',')

def getMode(path_train,path_valid,path_test):
    """
    Read the modes of training,valid and test data from the csv file 

    Parameters
    ----------
    path_train : string
        The path of the mode_train will be loaded
    path_valid : string
        The path of the mode_valid will be loaded
    path_test : string
        The path of the mode_test will be loaded
   
    Returns
    ----------
    mode_train: array
        Load the csv file to the array
    mode_valid: array
        Load the csv file to the array
    mode_test: array
        Load the csv file to the array
    """
    # Read the mode of training,valid and test dataset from the csv file
    mode_train = np.loadtxt(path_train,delimiter=",")
    mode_valid = np.loadtxt(path_valid,delimiter=",")
    mode_test = np.loadtxt(path_test,delimiter=",")
    return mode_train,mode_valid,mode_test



def PlotMSELoss(pathName,name):
    """
    This function is used to read the data to numpy form the specified path 
    and plot the mean square error(MSE) of training data and validation data 

    Parameters
    ----------
    pathName : string
        The path of the csv file that contains the information of epochs and 
        MSE of training data and validation data
    name : string
        The name of the Plot
    """
    epoch = pd.read_csv(pathName,usecols=[0]).values
    train_loss = pd.read_csv(pathName,usecols=[1]).values
    val_loss = pd.read_csv(pathName,usecols=[2]).values

    fig = plt.figure(figsize=(10,7))
    axe1 = plt.subplot(111)
    axe1.semilogy(epoch,train_loss,label = "train")
    axe1.plot(epoch,val_loss,label = "valid")
    axe1.legend(loc = "best",fontsize=14)
    axe1.set_xlabel("$epoch$",fontsize=14)
    axe1.set_ylabel("$MSE loss$",fontsize=14)
    axe1.set_title(name,fontsize=14)

def getTotal_decoded(training_decoded,valid_decoded,test_decoded,train_index,valid_index,test_index,nTotal=2000,nNodes=20550):
    """
    Becaue the training data, validation data and test data are randomly split. This
    function is used to Combine the training decoded, valid decoded and test decoded 
    to total decoded. The index of the total decoded is from 0 to 1999 in order

    Parameters
    ----------
    training_decoded : tensor
        The output of the autoencoder using the training data as the input
    valid_decoded : tensor
        The output of the autoencoder using the valid data as the input
    test_decoded : tensor
        The output of the autoencoder using the test data as the input
    train_index: array
        The array generated from the index_split function
    valid_index: array
        The array generated from the index_split function
    test_index: array
        The array generated from the index_split function
    nTotal: integer(optional)
        The value of the nTotal is fixed. This is the number of the solutions
    nNodes: integer(optional)
        The value of the nNodes is fixed. This is the number of the nodes of each solution
    Returns
    ----------
    total_decoded: array
        The output of the autoencoder using total data
    """
    total_decoded = np.zeros((nTotal,nNodes,2))
    for i in range(len(train_index)):
        total_decoded[int(train_index[i]),:,0] = training_decoded.cpu().detach().numpy()[i,:,0]
        total_decoded[int(train_index[i]),:,1] = training_decoded.cpu().detach().numpy()[i,:,1]

    for i in range(len(valid_index)):
        total_decoded[int(valid_index[i]),:,0] = valid_decoded.cpu().detach().numpy()[i,:,0]
        total_decoded[int(valid_index[i]),:,1] = valid_decoded.cpu().detach().numpy()[i,:,1]

    for i in range(len(test_index)):
        total_decoded[int(test_index[i]),:,0] = test_decoded.cpu().detach().numpy()[i,:,0]
        total_decoded[int(test_index[i]),:,1] = test_decoded.cpu().detach().numpy()[i,:,1]
    return total_decoded



def index_split(train_ratio, valid_ratio, test_ratio, total_num):
    """
    Random split the total_num according to the ratio train_ratio:valid_ratio : test_ratio.
    In this project, the train_ratio is 0.8, valid_ratio is 0.1 and test_ratio is 0.1. 

    Parameters
    ----------
    train_ratio : float
        Ratio of training data to total data
    valid_ratio : float
        Ratio of valid data to total data
    test_ratio : float
        Ratio of test data to total data
    total_num : integer
        This is the number of the total solution
   
    Returns
    ----------
    train_index: array
        The index of the training data
    valid_index: array
        The index of the valid data
    test_index: array
        The index of the test data
    """

    if train_ratio + valid_ratio + test_ratio != 1:
        raise ValueError("Three input ratio should sum to be 1!")
    total_index = np.arange(total_num)
    rng = np.random.default_rng()
    total_index = rng.permutation(total_index)
    knot_1 = int(total_num * train_ratio)
    knot_2 = int(total_num * valid_ratio) + knot_1
    train_index, valid_index, test_index = np.split(total_index, [knot_1, knot_2])
    return train_index, valid_index, test_index

def MSE(S,index, R, RT, nNodes=20550):
    """
    This function is used to calculate the mean square error of original  data and 
    reconstructed data using POD. This function is used to deal with flow past 
    cylinder data. When we use POD to deal with the burgers equation solution, there is
    a function called MSE as well. Although the basic theory is same, the details are
    different. Because these two functions are defined in two different ipynbs, the two
    functions with the same name are allowable.

    Parameters
    ----------
    S : array
        The array contains total data that combine two features as a column
    index : array
        It can be training index, valid index, test index or total index. 
        Depends on the MSE of which kind of data you want to calculate.
    R : array
        It contains the basis functions. The basis function can be calculated through 
        using singular value decomposition
    RT : array
        The transpose of the R
    nNodes: integer(optional)
        The value of the nNodes is fixed. This is the number of the nodes of each solution
    Returns
    ----------
    : float
        The result of the mean square error
    """

    m = np.zeros([2*nNodes,len(index)])
    n = np.zeros([len(index),nNodes,2])
    for i in range(len(index)):
        middle = RT@S[:,int(index[i])] # Add this one is used to reduce the computation cost
        m[:,i] = S[:,int(index[i])] - R @ middle
        n[i,:,0] = m[:nNodes,i]
        n[i,:,1] = m[nNodes:,i]
    return (n**2).mean()


# THis function is used to calculate the relative square error according to the above equation
def relative_error(S, index, R, RT,nNodes=20550):
    """
    This function is used to calculate the relative square error of original data and 
    reconstructed data using POD. This function is used to deal with flow past 
    cylinder data. When we use POD to deal with the burgers equation solution, there is
    a function called relative_error as well. Although the basic theory is same, the details are
    different. Because these two functions are defined in two different ipynbs, the two
    functions with the same name are allowable.

    Parameters
    ----------
    S : array
        The array contains total data that combine two features as a column
    index : array
        It can be training index, valid index, test index or total index. 
        Depends on the MSE of which kind of data you want to calculate.
    R : array
        It contains the basis functions. The basis function can be calculated through 
        using singular value decomposition
    RT : array
        The transpose of the R
    nNodes: integer(optional)
        The value of the nNodes is fixed. This is the number of the nodes of each solution
    Returns
    ----------
    : float
        The result of the relative square error
    """
    m = np.zeros([2*nNodes,len(index)])
    n = np.zeros([len(index),nNodes,2])
    h = np.zeros([len(index),nNodes,2])
    for i in range(len(index)):
        middle = RT@S[:,int(index[i])]  # Add this one is used to reduce the computation cost
        m[:,i] = S[:,int(index[i])] - R @ middle
        h[i,:,0] = S[nNodes:,int(index[i])]
        h[i,:,1] = S[:nNodes,int(index[i])]
        n[i,:,0] = m[:nNodes,i]
        n[i,:,1] = m[nNodes:,i]
    return (n**2).mean()/(h**2).mean()

def orginal(S,R,RT,nNodes=20550):
    """
    This function is used to calculate the reconstructed data from reduced variables using POD. 
    This function is used to deal with flow past cylinder data.

    Parameters
    ----------
    S : array
        The array contains total data that combine two features as a column
    index : array
        It can be training index, valid index, test index or total index. 
        Depends on the MSE of which kind of data you want to calculate.
    R : array
        It contains the basis functions. The basis function can be calculated through 
        using singular value decomposition
    RT : array
        The transpose of the R
    nNodes: integer(optional)
        The value of the nNodes is fixed. This is the number of the nodes of each solution
    Returns
    ----------
    orginal: array
        The reconstructed data from reduced variables using POD
    """
    N = len(S[1])
    orginal = np.zeros([2*nNodes,N])
    for i in range(N):
        middle = RT@S[:,i]   # Add this one is used to reduce the computation cost
        orginal[:,i] = R@middle
    return orginal


####################################Functions used in Burger equation####################################
def Save_pic(path,x,S,name):
    """
    This function is used to save all plots of the total data to a specified folder. 

    Parameters
    ----------
    path: string
        It is the path that the pictures will be saved
    x: array
       The x-axis value
    S : array
        The array contains total data in y-axis
    name: string
        The picture's name
    """

    Num = S.shape[1]
    for i in range(Num):
        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)
    ax1.plot(x,S[:,i])
    ax1.set_xlim(-100,100)
    ax1.set_ylim(0,1)
    plt.savefig(path + name+ "_BE_"+str(i)+".jpg")



def picTovideo(picPath,videoPath,Num,name):
    """
    This function is used to collect all pictures in a folder to generate a video 

    Parameters
    ----------
    picPath: string
        It is the path that the pictures that be saved
    videoPath: string
        It is the path that the video that will be saved
    Num : integer
        The number of the pictures
    name: string
        The picture's name
    """
    fps = 24  # The rate of the video. It can be changed
    frame = cv2.imread(picPath + name+"_BE_0.jpg")
    print(frame.shape)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(videoPath +'.avi',fourcc,fps,(frame.shape[1],frame.shape[0]),True)
    for i in range(Num):
        frame = cv2.imread(picPath+name+"_BE_"+str(i+1)+'.jpg')  # Read the pictures
        # cv2.imshow('frame',frame)
        videoWriter.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

def saveNumpy(path, array):
    """
    Save a array to csv

    Parameters
    ----------
    path: string
        It is the path that the  that will be saved
    array : array
    """
    np.savetxt(path, array, delimiter=',')


def getNumpy(path):
    """
    Load csv to a array

    Parameters
    ----------
    path: string
        It is the path that csv file has stored
    
    Returns
    ----------
    array : array
        The array store the csv file
 
    """
    array = np.loadtxt(path,delimiter=",")
    return array



def calculate_gamma(P,sigma):
    """
    Calcualte the fraction of the information

    Parameters
    ----------
    P: integer
        The number of the basis function
    sigma: array
        The array obtained from the singular value decomposition
    
    Returns
    ----------
    res : float
        Information carried by the first P basis function
    """

    M = len(sigma)
    sigma_sum = 0
    for i in range(M):
        sigma_sum += sigma[i]*sigma[i]
    P_sum = 0
    for i in range(P+1):
        P_sum += sigma[i]*sigma[i]
    res = P_sum / sigma_sum * 100  # 100 percent
    return res 



def MSE(S,index,R,RT,n=200):
    """
    This function is used to calculate the mean square error of original data and 
    reconstructed data using POD. This function is used to deal with burgers equation data. 
    When we use POD to deal with the flow past cylinder solution, there is
    a function called MSE as well. Although the basic theory is same, the details are
    different. Because these two functions are defined in two different ipynbs, the two
    functions with the same name are allowable.

    Parameters
    ----------
    S : array
        The array contains total data
    index : array
        It can be training index, valid index, test index or total index. 
        Depends on the MSE of which kind of data you want to calculate.
    R : array
        It contains the basis functions. The basis function can be calculated through 
        using singular value decomposition
    RT : array
        The transpose of the R
    n: integer(optional)
        The value of the n is fixed. This is the number of the nodes of each solution
    Returns
    ----------
    : float
        The result of the mean square error
    """
    k = np.zeros([n,len(index)])
    for i in range(len(index)):
        k[:,i] = S[:,int(index[i])] - R @ RT @ S[:,int(index[i])]
    return (k**2).mean()


def relative_error(S,index,R,RT,n=200):
    """
    This function is used to calculate the relative square error of original data and 
    reconstructed data using POD. This function is used to deal with burgers equation data. 
    When we use POD to deal with the flow past cylinder solution, there is
    a function called relative_error as well. Although the basic theory is same, the details are
    different. Because these two functions are defined in two different ipynbs, the two
    functions with the same name are allowable.

    Parameters
    ----------
    S : array
        The array contains total data
    index : array
        It can be training index, valid index, test index or total index. 
        Depends on the MSE of which kind of data you want to calculate.
    R : array
        It contains the basis functions. The basis function can be calculated through 
        using singular value decomposition
    RT : array
        The transpose of the R
    n: integer(optional)
        The value of the n is fixed. This is the number of the nodes of each solution
    Returns
    ----------
    : float
        The result of the relative square error
    """
    k = np.zeros([n,len(index)])
    h = np.zeros([n,len(index)])
    for i in range(len(index)):
        k[:,i] = S[:,int(index[i])] - R @ RT @ S[:,int(index[i])]
        h[:,i] = S[:,int(index[i])]
    return (k**2).mean()/(h**2).mean()


def SVD_MSE(S,decoded,R,n=200):
    """
    This function is used to calculate the MSE of SVD_AE in burgers equation

    Parameters
    ----------
    S : array
        The array contains total data
    decoded : array
        The output of the autoencoder
    R : array
        It contains the basis functions. The basis function can be calculated through 
        using singular value decomposition
    n: integer(optional)
        The value of the n is fixed. This is the number of the nodes of each solution
    Returns
    ----------
    : float
        The result of MSE
    """
    decoded1 = decoded.transpose()
    k = np.zeros([n,S.shape[1]])
    for i in range(S.shape[1]):
        k[:,i] = S[:,i] - R @ decoded1[:,i]
    return (k**2).mean()
