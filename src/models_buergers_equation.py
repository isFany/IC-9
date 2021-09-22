"""
The file contains models that will be used in the burgers equation. This file is only used to add the notes
and generate the documentation. These models have been added to the ipynb files

@author: Fan Yang
"""
import torch  # Pytorch
import torch.nn as nn  # Neural network module
import torch.nn.functional as fn  # Function module
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, TensorDataset, Dataset

#################################### FC-AE in Bugers equation ####################################

class FC(nn.Module):
    """
    The architecture of the fully-connected autoencoder is used in burgers equation. It consists of 
    fully connected layers.

    Parameters
    ----------
    hidden_1 : int
        The number of the latent variable can be adjusted from 1 to any proper value
    """
    def __init__(self,hidden_1):
        super(FC, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(200, hidden_1),
            nn.ReLU(),
            # nn.Sigmoid(),
           
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_1, 200),
            nn.ReLU(),
            # nn.Sigmoid(),
        )


    def forward(self,x):
        """
        Send the data to the encoder and decoder in order to get the corresponding output

        Parameters
        ----------
        x : tensor or numpy
            The input data to the autoencoder

        Returns
        -------
        encoded : tensor
            The output of the encoder in the fully-connected autoencoder
        decoded : tensor
            The output of the decoder in the fully-connected autoencoder
        """
        encoded = self.fc1(x)
        decoded = self.fc2(encoded)
        return encoded, decoded



def weight_init(m):
    """
    This method is used to initialise the model's weight and bias. For the 
    linear layer, we will use xavier initialization

    Parameters
    ----------
    m : model
        Newly created autoencoder
    """
    classname = m.__class__.__name__
    if classname.find("Linear")!=-1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias, 0.05)


#################################### SVD-AE in Bugers equation ####################################
class SVD_FC(nn.Module):
    """
    The architecture of SVD autoencoder is used in burgers equation. It consists of 
    fully connected layers. We have taken SVD to reduce the nodes from 200 to 150
    and use this autoencoder to further reduce the variables. 

    Parameters
    ----------
    hidden_1 : int
        The number of the latent variable can be adjusted from 1 to any proper value
    """
    def __init__(self,hidden_1):
        super(SVD_FC, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(150, hidden_1),
            nn.Sigmoid(),   
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_1, 150),
            nn.Sigmoid(),
        )


    def forward(self,x):
        """
        Send the data to the encoder and decoder in order to get the corresponding output

        Parameters
        ----------
        x : tensor or numpy
            The input data to the autoencoder

        Returns
        -------
        encoded : tensor
            The output of the encoder in the SVD autoencoder
        decoded : tensor
            The output of the decoder in the SVD autoencoder
        """
        encoded = self.fc1(x)
        decoded = self.fc2(encoded)
        return encoded, decoded

#################################### CAE in Bugers equation ####################################
# The architecture of the first subnetwork
class CAE_1(nn.Module):
    """
    The architecture of the convolutional autoencoder is used in burgers equation. It consists of 
    convolutional layers and fully connected layers.

    Parameters
    ----------
    hidden_1 : int
        The number of the latent variable can be adjusted from 1 to any proper value
    """
    def __init__(self,hidden_1):
        super(CAE_1,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,8,3,3,15),
            nn.ReLU(),
            nn.Conv1d(8,16,3,3,15),
            nn.ReLU(),
            nn.Conv1d(16,16,3,3,15),
            nn.ReLU(),
        )
        self.fc1=nn.Sequential(
            nn.Linear(16*21,hidden_1),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_1,16*21),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16,16,3,3,14),
            nn.ReLU(),
            nn.ConvTranspose1d(16,8,4,3,15),
            nn.ReLU(),
            nn.ConvTranspose1d(8,1,3,3,14),
            nn.ReLU(),
        )

    def forward(self,x):
        """
        Send the data to the encoder and decoder in order to get the corresponding output

        Parameters
        ----------
        x : tensor or numpy
            The input data to the autoencoder

        Returns
        -------
        encoded : tensor
            The output of the encoder in the convolutional autoencoder
        decoded : tensor
            The output of the decoder in the convolutional autoencoder
        """
        encoded_1 = self.encoder(x.view(-1,1,200))
        encoded = self.fc1(encoded_1.view(-1,16*21))
        decoded_1 = self.fc2(encoded)
        decoded = self.decoder(decoded_1.view(-1,16,21)).view(-1,200)
        
        return encoded,decoded

def weights_init(m):
    """
    This method  is used to initialise the convolutional autoencoder's weight 
    and bias. For the linear layer, we will use xavier initialization. For 
    the Conv1d and ConvTranspose1d layer, we will use kaiming initialization.

    Parameters
    ----------
    m : model
        Newly created autoencoder
    """
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        # Apply kaiming initialization to Conv1d layer
        nn.init.constant_(m.bias.data, 0)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        # Apply xavier initialization to Linear layer
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias.data, 0.05)
    elif classname.find('ConvTranspose1d') != -1:
        # Apply kaiming initialization to ConvTranspose1d layer
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0.05)

#################################### HAE in Bugers equation ####################################
# The second network architecture 
class CAE_2(nn.Module):
    """
    The architecture of the second subnetwork in hierarchical autoencoder is used 
    in burgers equation. It consists of convolutional layers and fully connected layers.

    Parameters
    ----------
    hidden_2 : int
        The number of the latent variable can be adjusted from 1 to any proper value
    """
    def __init__(self,hidden_2):
        super(CAE_2,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,16,3,2,5),  # (b,16,104)
            nn.ReLU(),
            nn.Conv1d(16,16,3,2,5),  #(b,16,56)
            nn.ReLU(),
            nn.Conv1d(16,16,3,2,5),  #(b,16,32)
            nn.ReLU(),
        )
        self.fc1=nn.Sequential(
            nn.Linear(16*32,hidden_2),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2*hidden_2,16*32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16,16,4,2,5),  #(b,16,56)
            nn.ReLU(),
            nn.ConvTranspose1d(16,16,4,2,5),  #(b,16,104)
            nn.ReLU(),
            nn.ConvTranspose1d(16,1,4,2,5),  #(b,1,200)
            nn.ReLU(),
        )

    def forward(self,x,mode):
        """
        Send the data to the encoder and decoder in order to get the corresponding output

        Parameters
        ----------
        x : tensor or numpy
            The input data to the autoencoder
        
        mode: numpy
            The output of the first subnetwork's encoder

        Returns
        -------
        encoded : tensor
            The output of the encoder in the convolutional autoencoder
        decoded : tensor
            The output of the decoder in the convolutional autoencoder
        """
        encoded_1 = self.encoder(x.view(-1,1,200))
        encoded = self.fc1(encoded_1.view(-1,16*32))
        encoded = torch.cat((encoded, mode),axis = 1)
        decoded_1 = self.fc2(encoded)
        decoded = self.decoder(decoded_1.view(-1,16,32)).view(-1,200)
        
        return encoded,decoded
