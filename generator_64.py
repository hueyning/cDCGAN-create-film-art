# Generator class for 64 x 64 images

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