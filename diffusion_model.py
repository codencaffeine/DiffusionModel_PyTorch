    #* This line checks whether this script is run as a main program and not imported as a module
    if __name__ == "__main__":
        #* Dict and Tuple imported from typing module is used as type hints in the code for better readability 
        from typing import Dict, Tuple
        #* To visualize the progress during any iterative process or loops as a progress bar, we can use the tqdm library 
        from tqdm import tqdm
        #* We will import the torch module, which is the core module of PyTorch. It provides the fundamental components 
        #* for creating and working with tensors, defining neural networks, and performing various mathematical operations used in deep learning
        import torch
        #* The torch.nn module helps us define different neural networks and to define layers in them. It provides base classes to build custom 
        #* neural networks like sklearn, and also to have predefined layers like convulutional layers, linear layers, activation layers, etc
        import torch.nn as nn
        """
        This module provides with functions and are stateless, meaning, they dont have any internal parameters(weights and biases) to be learned 
        and will just compute a something when called. You can understand it in a simple way as follows: torch.nn contains classes that represents layers
        in a neural network, which torch.nn.functionality contains basic functions, which provides more flexibility.
        """
        import torch.nn.functional as F
        #* The DataLoader heps in loading data in batches durng the training process
        from torch.utils.data import DataLoader
        #* We import models and transforms from torchvision, to have predefined models and image transformations
        from torchvision import models, transforms
        #* These functions helps to save images generated and to make grids of images
        from torchvision.utils import save_image, make_grid
        #* This library is used for visualization using plots
        import matplotlib.pyplot as plt
        #* The FuncAnimation and PillowWriter classes from animation module of Matplotlib is used for creating animations and saving them
        from matplotlib.animation import FuncAnimation, PillowWriter
        #* This library is used for numerical operations and array manipulations
        import numpy as np
        #* Importing the HTML class from IPython's display module, which is used to display HTML content in IPython environments like Jupyter Notebook.
        from IPython.display import HTML
        # from diffusion_utilities import *
        #---------------------------------------------------------------------------------------------------------------------------------------------
        """
        We are building a custom PyTorch module named ContextUnet and using the nn.Module as the base class. We are doing this 
        to build a U-Net based model for our diffusion model. 
        """
        class ContextUnet(nn.Module):
            #* We now initialize the class using the __init__ constructor function
            #* The parameters this function take are as follows:
            #* in_channels: The number of input channels in the data
            #* n_feat: The number of intermediate feature maps
            #* n_cfeat: Number of context features
            #* height: The height of the input images
            def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):
                #* Here we are calling the constructor of the parent class nn.Module.
                #* This is to ensure that the base class is initialized properly before we set up ur custom model
                super(ContextUnet, self).__init__()
                # number of input channels, number of intermediate feature maps and number of classes
                """
                No we will store the input arguments(in_channel, n_feat, n_cfeat, height) as instance variables so 
                that all the methods defined within this class can access them.
                """
                self.in_channels = in_channels
                self.n_feat = n_feat
                self.n_cfeat = n_cfeat
                self.h = height  #assume h == w. must be divisible by 4, so 28,24,20,16...
                """
                Now we initialize the first layer of our network, which is a combination of convolutional layer and a residual connection
                Thus ResidualConvBlock is a custom building block we will be using to enhance the model's capacity and its performance.
                A brief overview of what this block is used for:
                A convolutional layer applies convolutional operations to the input data, which helps the model to learn spatial patterns 
                and features in the data. A residual connection, also known as skip connection, is a technique used to address the vanishing 
                gradient problem in very deep neural networks.
                How a residual connection works is, it adds the original input or a modified version of it, to the output of the convolutional 
                layer, so that the network can learn the difference or the residual between the input and the mapped output. This improves the 
                capacity of the model to capture complex and abstract representations of the input data and deals with th vanishing gradient problem. 
                """
                #* The parameters are in_channels: Number of channels(3 if RGB), n_feat: Number of intermediate feature maps, 
                #* and is_res: Boolean value to indicate whether to use residual connection or not   
                self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
                #* Initialize the down-sampling path of the U-Net with two levels
                #* The down1 and down2 blocks perform downsampling with convolutional layers to build the down-sampling path of the
                """
                UnetDown takes two parameters: 
                in_channels: the number of input channels, which is equal to the number of output channels from the 
                previous layer (n_feat in this case)
                out_channels: The number of output channels for the down-sampling path.
                The U-Net downblock applies a convolutional layer with the specified number of input and output channels. This helps to reduce the spatial 
                dimensions and learn relevant features. It performs downsampling to reduce the spatial resolution of the feature mapsU-Net
                The level 2 (self.down2) is similar to the previous line, but the number of output channels is doubled (2 * n_feat), which increases the number of feature maps 
                learned in the down-sampling path. This leads to a higher level of abstraction as the model goes deeper
                """
                self.down1 = UnetDown(n_feat, n_feat)        # down1 #[10, 256, 8, 8]
                self.down2 = UnetDown(n_feat, 2 * n_feat)    # down2 #[10, 256, 4,  4]
                
                # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
                #* This part is This block is responsible for converting the feature maps obtained from the down-sampling path into a vector representation. It prepares the feature 
                #* maps to be used in the later part of the network.
                #* self.to_vec is a sequential container in PyTorch that holds a sequence of operations that will be applied in order
                #* This operation performs average pooling over the spatial dimensions of the feature maps using a kernel size of 4x4. Average pooling downsamples the spatial resolution 
                #* of the feature maps, aggregating information within each 4x4 neighborhood and reducing the spatial dimensions.
                #* To introduce non-linearity to the model, we apply the GELU activation function to the output of the average pooling operation. This operation
                #* will be performed element-wise on the output of the previous operation.
                self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

                #* Now we will embed the timestep and context labels using a one-layer fully connected neural network EmbedFC
                #* These embeddings are important to capture the temporal and contextual information in the data and to condition the model on the context labels.
                #* is a custom class that creates a one-layer fully connected neural network for embedding the input data.
                #* in_features: The number of input features (in this case, it is 1, representing the timestep) is the first argument this class takes in
                #* out_features: The number of output features (in this case, it is 2*n_feat) is the second argument this class takes in, it doubles the number of intermediate feature maps 
                #* (n_feat) to create more expressive embeddings.
                self.timeembed1 = EmbedFC(1, 2*n_feat)
                #* This is similar to the previous line, but the number of output features is equal to n_feat. It creates embeddings with the same number of feature maps as the intermediate 
                #* feature maps (n_feat)
                self.timeembed2 = EmbedFC(1, 1*n_feat)
                #* This is similar to self.timeembed1, but it embeds the context labels. The number of input features is equal to n_cfeat, which is the size of the context vector (n_cfeat)
                #* The number of output features is 2 * n_feat, doubling the number of intermediate feature maps.
                self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
                #*  This embeds the context labels with the same number of feature maps as the intermediate feature maps (n_feat)
                self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

                # Initialize the up-sampling path of the U-Net with three levels
                #* self.up0 is a sequential container that holds a sequence of operations that will be applied in order
                #* This operation performs transposed convolution on the input feature maps with the specified number of input and output channels. This helps to increase the spatial dimensions
                #* and learn relevant features. It performs upsampling to increase the spatial resolution of the feature maps.
                self.up0 = nn.Sequential(
                    #* This is a transposed convolutional layer that performs upsampling of the feature maps. It takes the number of input and output channels as 2 * n_feat, and the kernel size 
                    #* is determined by self.h // 4, which upsamples the feature maps by a factor of 4 in both height and width.
                    nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), 
                    #* Group normalization is applied to normalize the output of the transposed convolutional layer. This step is crucial to avoid overfitting and to improve the performance of the model.
                    nn.GroupNorm(8, 2 * n_feat), # normalize  
                    #* ReLU activation function is applied element-wise to introduce non-linearity.                     
                    nn.ReLU(),
                )
                #* The self.up1 and self.up2 are instances of the custom UnetUp class. These blocks perform up-sampling and concatenation with the corresponding feature maps from the down-sampling path.
                self.up1 = UnetUp(4 * n_feat, n_feat)
                self.up2 = UnetUp(2 * n_feat, n_feat)

                #* Now we will initialize the final convolutional layers to map to the same number of channels as the input image
                #* Again a sequential container of PyTorch that holds a sequence of operations that will be applied in order
                self.out = nn.Sequential(
                    #* This is a convolutional layer that reduces the number of feature maps from 2 * n_feat to n_feat. It uses a kernel size of 3x3, stride of 1, and padding of 1 to maintain the spatial dimensions.
                    nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
                    #* Group normalization is applied to normalize the output of the convolutional layer.
                    nn.GroupNorm(8, n_feat), # normalize
                    #* Introduce non-linearity using the ReLU activation function.
                    nn.ReLU(),
                    #* This is a convolutional layer that maps the feature maps to the same number of channels as the input image. It uses a kernel size of 3x3, stride of 1, and padding of 1 to maintain the spatial dimensions.
                    nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map to same number of channels as input
                )
            #* The forward method dfines the forward path of the U-Net
            def forward(self, x, t, c=None):
                """
                x : (batch, n_feat, h, w) : input image shape (batch, 1, 64, 64)
                t : (batch, n_cfeat)      : time step with shape (batch, 1)
                c : (batch, n_classes)    : context label with shape (batch, 1)
                """
                #* x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

                #* We pass the input image through the initial convolutional layer
                x = self.init_conv(x)
                #* Then we pass the result through the down-sampling path
                down1 = self.down1(x)       #[10, 256, 8, 8]
                down2 = self.down2(down1)   #[10, 256, 4, 4]
                
                #* We then convert the feature maps to a vector and apply an activation
                hiddenvec = self.to_vec(down2)
                
                #* If the context label "c" is not provided, we initialize it to a vector of zeros
                if c is None:
                    c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
                    
                #* We then embed context and timestep using the defined embedding layers as follows 
                cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)     # (batch, 2*n_feat, 1,1)
                temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
                cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
                temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
                #print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")

                #* Embed the context and timestep using the defined embedding layers, perform up-sampling in the up-sampling 
                #* path, adding and multiplying the embeddings
                up1 = self.up0(hiddenvec)
                up2 = self.up1(cemb1*up1 + temb1, down2)  # add and multiply embeddings
                up3 = self.up2(cemb2*up2 + temb2, down1)
                #* We then pass the concatenated feature maps through the final convolutional layers (self.out) to obtain the 
                #* output of the model (out).
                out = self.out(torch.cat((up3, x), 1))
                return out
        # hyperparameters

        # diffusion hyperparameters
        #* timesteps is the number of diffusion timesteps used during sampling. Here, we set to 500.
        timesteps = 500
        #* beta1 and beta2 are the parameters used in the diffusion process. They control the variance of the noise added at each timestep.
        beta1 = 1e-4
        beta2 = 0.02

        # network hyperparameters
        #* Now we check if CUDA (GPU) is available and sets the device accordingly. If not, it uses the CPU.
        device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
        #* The number of hidden dimensions in the feature maps of the U-Net model is set to 64.
        n_feat = 64 
        #* The size of the context vector is set to 5.
        n_cfeat = 5 
        #* The height of the input images is set to 16 for 16x16 images.
        height = 16 
        #* We save the model weights in the weights folder or directory.
        save_dir = './weights/'

        # training hyperparameters
        #* The number of samples in each batch during training
        batch_size = 100
        #* The number of training epochs
        n_epoch = 32
        #* The learning rate for optimization during training
        lrate=1e-3
        # construct DDPM noise schedule
        #* We construct the noise schedule for the DDPM process using the beta1 and beta2 parameters and the number of timesteps.
        #* b_t represents the diffusion beta values linearly spaced from beta1 to beta2. It is computed using torch.linspace() and has timesteps + 1 elements
        b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
        #* a_t is the complement of b_t, computed as 1 - b_t. It represents the alphas in the diffusion process
        a_t = 1 - b_t
        #* It is the cumulative sum of the log of a_t along the time axis. This sequence is used to scale the noise during the denoising process
        ab_t = torch.cumsum(a_t.log(), dim=0).exp() 
        ab_t[0] = 1
        
        # Construction of the model
        #* Create an instance of the U-Net model and send it to the device (CPU or GPU) for training. The number of channels is 3 since RGB images are used.
        nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
        #* Now we will define sampling function for DDIM (Denoising Diffusion Probabilistic Models) using the trained model 
        #* It removes the noise using ddim method
        def denoise_ddim(x, t, t_prev, pred_noise):
            ab = ab_t[t]
            ab_prev = ab_t[t_prev]
            
            x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
            dir_xt = (1 - ab_prev).sqrt() * pred_noise

            return x0_pred + dir_xt
        # load in model weights and set to eval mode
        nn_model.load_state_dict(torch.load(f"{save_dir}/model_31.pth", map_location=device))
        nn_model.eval() 
        print("Loaded in Model without context")
        # sample quickly using DDIM
        @torch.no_grad()
        def sample_ddim(n_sample, n=20):
            # x_T ~ N(0, 1), sample initial noise
            samples = torch.randn(n_sample, 3, height, height).to(device)  

            # array to keep track of generated steps for plotting
            intermediate = [] 
            step_size = timesteps // n
            for i in range(timesteps, 0, -step_size):
                print(f'sampling timestep {i:3d}', end='\r')

                # reshape time tensor
                t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

                eps = nn_model(samples, t)    # predict noise e_(x_t,t)
                samples = denoise_ddim(samples, i, i - step_size, eps)
                intermediate.append(samples.detach().cpu().numpy())

            intermediate = np.stack(intermediate)
            return samples, intermediate
        # visualize samples
        plt.clf()
        samples, intermediate = sample_ddim(32, n=25)
        animation_ddim = plot_sample(intermediate,32,4,save_dir, "ani_run", None, save=False)
        HTML(animation_ddim.to_jshtml())
        # load in model weights and set to eval mode
        nn_model.load_state_dict(torch.load(f"{save_dir}/context_model_31.pth", map_location=device))
        nn_model.eval() 
        print("Loaded in Context Model")
        # fast sampling algorithm with context
        @torch.no_grad()
        def sample_ddim_context(n_sample, context, n=20):
            # x_T ~ N(0, 1), sample initial noise
            samples = torch.randn(n_sample, 3, height, height).to(device)  

            # array to keep track of generated steps for plotting
            intermediate = [] 
            step_size = timesteps // n
            for i in range(timesteps, 0, -step_size):
                print(f'sampling timestep {i:3d}', end='\r')

                # reshape time tensor
                t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

                eps = nn_model(samples, t, c=context)    # predict noise e_(x_t,t)
                samples = denoise_ddim(samples, i, i - step_size, eps)
                intermediate.append(samples.detach().cpu().numpy())

            intermediate = np.stack(intermediate)
            return samples, intermediate
        # visualize samples
        plt.clf()
        ctx = F.one_hot(torch.randint(0, 5, (32,)), 5).to(device=device).float()
        samples, intermediate = sample_ddim_context(32, ctx)
        animation_ddpm_context = plot_sample(intermediate,32,4,save_dir, "ani_run", None, save=False)
        HTML(animation_ddpm_context.to_jshtml())
        # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
        def denoise_add_noise(x, t, pred_noise, z=None):
            if z is None:
                z = torch.randn_like(x)
            noise = b_t.sqrt()[t] * z
            mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
            return mean + noise
        # sample using standard algorithm
        @torch.no_grad()
        def sample_ddpm(n_sample, save_rate=20):
            # x_T ~ N(0, 1), sample initial noise
            samples = torch.randn(n_sample, 3, height, height).to(device)  

            # array to keep track of generated steps for plotting
            intermediate = [] 
            for i in range(timesteps, 0, -1):
                print(f'sampling timestep {i:3d}', end='\r')

                # reshape time tensor
                t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

                # sample some random noise to inject back in. For i = 1, don't add back in noise
                z = torch.randn_like(samples) if i > 1 else 0

                eps = nn_model(samples, t)    # predict noise e_(x_t,t)
                samples = denoise_add_noise(samples, i, eps, z)
                if i % save_rate ==0 or i==timesteps or i<8:
                    intermediate.append(samples.detach().cpu().numpy())

            intermediate = np.stack(intermediate)
            return samples, intermediate
        %timeit -r 1 sample_ddim(32, n=25)
        %timeit -r 1 sample_ddpm(32, )

#Code credits: DeepLearning.AI course on Generative Models