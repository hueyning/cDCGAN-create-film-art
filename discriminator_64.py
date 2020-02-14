# discriminator class for 64 x 64 images

from modules import *

# discriminator code
class Discriminator_64(nn.Module):
    
    def __init__(self, n_classes, ngpu, nc = 3, ndf = 64):
        '''
        Discriminator for images with dimensions: 64 x 64 x 3.
        
        Inputs
            n_classes (INT)
                Number of classes in dataset.
            
            ngpu (INT)
                Number of GPUs to be used in training process.
                
            ndf (INT)
                Size of feature maps in discriminator.
                Default: 64.
        '''
        
        super(Discriminator_64, self).__init__()
        
        self.ngpu = ngpu
        
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        
        self.convolution_layers = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
        
        self.linear_layers = nn.Sequential(
            
            nn.Linear(in_features = 1 + n_classes, # flattened output from last conv + embedding
                      out_features = 512), # arbitrary + based on external references
            
            nn.LeakyReLU(0.2, inplace=True) ,
        
            nn.Linear(in_features = 512, # output from last linear layer
                      out_features = 1), # true or false image
            
            nn.Sigmoid()
        )
        

    def forward(self, input, labels):
        
        x = self.convolution_layers(input) # run input through convolutional layers
        # print(x.shape) # output shape: (128,1,1,1)
        x = x.view(x.size(0), -1) # flatten output from main
        # print(x.shape) # output shape: (128,1)
        y = self.label_embedding(labels) # create label layer
        # print(y.shape) # output shape: (128,3)
        x = torch.cat((x, y), -1) # concatenate flattened output to label layer
        # print(x.shape) # output shape: (128,4)
        x = self.linear_layers(x) # run flattened + merged layer through linear layers
        
        return x