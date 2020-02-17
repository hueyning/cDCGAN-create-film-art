# Generator and Discriminator class for 64 x 64 x 3 images

from modules import *

# Generator Code
class Generator_64(nn.Module):
    
    def __init__(self, n_classes, ngpu, nz = 100, nc = 3, ngf = 64):
        '''
        Generator for images with dimensions: 64 x 64 x 3.
        
        Inputs
            n_classes (INT)
                Number of classes in dataset.
            
            ngpu (INT)
                Number of GPUs to be used in training process.
                
            ngf (INT)
                Size of feature maps in generator.
                Default: 64.
        '''
        
        super(Generator_64, self).__init__()
        
        self.nz = nz

        self.ngpu = ngpu
        
        self.n_classes = n_classes
        
        self.label_emb = nn.Embedding(n_classes, n_classes)
        
        self.main = nn.Sequential(
            
            # input is Z + n_classes, going into a convolution
            nn.ConvTranspose2d(nz + n_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, labels):
    
        # Concatenate label embedding and noise to produce input
        flat_embed_input = torch.cat((self.label_emb(labels), input), -1)

        # reshape flattened layer to torch.Size([128, nz + n_classes, 1, 1])
        reshaped_input = flat_embed_input.view((-1, self.nz + self.n_classes,1,1)) 
        
        gen_img = self.main(reshaped_input)
        
        return gen_img


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