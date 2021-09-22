"""
The file is used to convert csv file to vtu file. It is provided by Dr Claire Heaney.
"""



import numpy as np
import vtktools
import os 


def get_clean_vtu(filename):
    """
    This function is used to remove fields and arrays from a vtk file, leaving the 
    coordinates/connectivity information.

    Parameters
    ----------
    filename : string
        The name of the vtu file
        
    """
    vtu_data = vtktools.vtu(filename)
    clean_vtu = vtktools.vtu()
    clean_vtu.ugrid.DeepCopy(vtu_data.ugrid)
    fieldNames = clean_vtu.GetFieldNames()
# remove all fields and arrays from this vtu
    for field in fieldNames:
        clean_vtu.RemoveField(field)
        fieldNames = clean_vtu.GetFieldNames()
        vtkdata=clean_vtu.ugrid.GetCellData()
        arrayNames = [vtkdata.GetArrayName(i) for i in range(vtkdata.GetNumberOfArrays())]
    for array in arrayNames:
        vtkdata.RemoveArray(array)
    return clean_vtu


latent_num = 2
Batch_size = 16

folder_name = "SVDAE"+"_"+str(latent_num)
vtufile_name = folder_name + '/SVDAE_reconstructed_'
# folder_name = "SVDAE"+"_LV"+str(latent_num)+"_error"
# vtufile_name = folder_name + '/SVDAE_error_'



#cwd = os.getcwd()
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)  
#os.chdir('csv_data') # will overwrite files in results

# This path can be changed accoridng to the users requirement
csv_data = np.loadtxt('.\All_result\SVDAE_II_LV2_B16E_2000_result.csv', delimiter=',')
print('shape csv_data', csv_data.shape)

nExamples = csv_data.shape[0]
nNodes = 20550
nDim = 2 # physical dimension
print ('nNodes*nDim',nNodes*nDim)
assert csv_data.shape[1]-nNodes*nDim == 0, "results was not the shape you were expecting"

# get clean vtu file - path to original vtu data
path = './FPC_Re3900_DG_new'
filename = path + '/fpc_0.vtu'

clean_vtu = get_clean_vtu(filename)
velocity = np.zeros((nNodes,3)) # whether 2D or 3D

for i in range(nExamples):
    new_vtu = vtktools.vtu()
    new_vtu.ugrid.DeepCopy(clean_vtu.ugrid)
    new_vtu.filename = vtufile_name+ str(i) + '.vtu'
 
    velocity[:,0:nDim] = csv_data[i,:].reshape((nNodes,nDim),order='F')
    new_vtu.AddField('Velocity_CAE',velocity)
    new_vtu.Write()