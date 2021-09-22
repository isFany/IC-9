"""
This file is used to get the velocity at a fixed point in vtu file. It is provided by Dr. Claire Heaney.
"""



import vtk, vtktools
import sys, os
import numpy as np
import matplotlib.pyplot as plt

coordinates = np.array([[0.4,0.15,0],[0.4,0.2,0],[0.4,0.25,0]])
N = 1000 # time levels
Npred = 0 # prediction starts at 100
print(coordinates.shape[1]+1)
values_at_sensors = np.zeros((N,coordinates.shape[1]+1))  
values_at_sensors[:,0] = np.linspace(0,N-1,N)

prediction_at_CAE = np.zeros((N,coordinates.shape[1]+1))   
prediction_at_CAE[:,0] = np.linspace(0,N-1,N)

prediction_at_HAE = np.zeros((N,coordinates.shape[1]+1))   
prediction_at_HAE[:,0] = np.linspace(0,N-1,N)

prediction_at_SAE = np.zeros((N,coordinates.shape[1]+1))   
prediction_at_SAE[:,0] = np.linspace(0,N-1,N)

prediction_at_SVD = np.zeros((N,coordinates.shape[1]+1))   
prediction_at_SVD[:,0] = np.linspace(0,N-1,N)

prediction_at_SVDAE = np.zeros((N,coordinates.shape[1]+1))   
prediction_at_SVDAE[:,0] = np.linspace(0,N-1,N)

for k in range(N):

    filename = "./FPC_Re3900_DG_new_sum/FPC_Re3900_DG_new/fpc_" + str(k) + ".vtu"
    vtu_data = vtktools.vtu(filename)
    v = vtu_data.ProbeData(coordinates, "Velocity")
    norm_v = np.linalg.norm(v,axis=1)
    values_at_sensors[k,1:] = norm_v

    # filename = "./CAE_64/CAE_reconstructed_" + str(k) + ".vtu"
    # vtu_data = vtktools.vtu(filename)
    # v = vtu_data.ProbeData(coordinates, "Velocity_CAE")
    # norm_v = np.linalg.norm(v,axis=1)
    # prediction_at_CAE[k,1:] = norm_v

    # filename = "./HAE_64/reconstructed_" + str(k) + ".vtu"
    # vtu_data = vtktools.vtu(filename)
    # v = vtu_data.ProbeData(coordinates, "Velocity_CAE")
    # norm_v = np.linalg.norm(v,axis=1)
    # prediction_at_HAE[k,1:] = norm_v

    filename = "./SAE_64/SAE_reconstructed_" + str(k) + ".vtu"
    vtu_data = vtktools.vtu(filename)
    v = vtu_data.ProbeData(coordinates, "Velocity_CAE")
    norm_v = np.linalg.norm(v,axis=1)
    prediction_at_SAE[k,1:] = norm_v

    # filename = "./SVD_LV64/SVD_reconstructed_" + str(k) + ".vtu"
    # vtu_data = vtktools.vtu(filename)
    # v = vtu_data.ProbeData(coordinates, "Velocity_CAE")
    # norm_v = np.linalg.norm(v,axis=1)
    # prediction_at_SVD[k,1:] = norm_v

    # filename = "./SVDAE_64/SVDAE_reconstructed_" + str(k) + ".vtu"
    # vtu_data = vtktools.vtu(filename)
    # v = vtu_data.ProbeData(coordinates, "Velocity_CAE")
    # norm_v = np.linalg.norm(v,axis=1)
    # prediction_at_SVDAE[k,1:] = norm_v

    

#norm_v = np.atleast_2d( np.linalg.norm(v,axis=1)).T
#print norm_v

plt.rcParams['figure.figsize'] = (8.0, 4.0) # Set the figure size
plt.rcParams['savefig.dpi'] = 300 # Set the figure pixel
plt.rcParams['figure.dpi'] = 300

fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(111)
ax1.set_xlim(0,1000)
ax1.set_ylim(0,0.08)

plt.tick_params(labelsize=15)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

ax1.plot(values_at_sensors[:,1], linewidth=2.0,label='Original')
# ax1.plot(prediction_at_CAE[Npred:,0],prediction_at_CAE[Npred:,1], linewidth=2.0,label='SFC-CAE')
# ax1.plot(prediction_at_HAE[Npred:,0],prediction_at_HAE[Npred:,1], linewidth=2.0,label='SFC-HAE')
ax1.plot(prediction_at_SAE[Npred:,0],prediction_at_SAE[Npred:,1], linewidth=2.0,label='SFC-SAE')
# ax1.plot(prediction_at_SVD[Npred:,0],prediction_at_SVD[Npred:,1], linewidth=2.0,label='POD')
# ax1.plot(prediction_at_SVDAE[Npred:,0],prediction_at_SVDAE[Npred:,1], linewidth=2.0,label='SVD-AE')
ax1.legend(loc = "best",fontsize=16)
ax1.set_xlabel('$time step$', fontsize=20)
ax1.set_ylabel('$value$', fontsize=20)
plt.savefig('SAE_velocity.png',bbox_inches='tight')



