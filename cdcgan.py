# Complete cDCGAN class

from modules import *

class cDCGAN(object):
    
    '''
    Conditional DCGAN class.
    '''
    
    def _checkDirectory(self, dirName):
        
        if not os.path.exists(dirName):
            print(f"{dirName} directory does not exist. Making {dirName}")
            os.makedirs(dirName)
            
        else: print(f"{dirName} directory exists.")
    
    
    def __init__(self, dataloader, classes, save_dir, num_epochs,
                 criterion, netD, netG, optimizerD, optimizerG, device, nz = 100):
        
        # data parameters
        self.dataloader = dataloader
        self.classes = classes # class labels
        self.n_classes = len(classes) # number of classes
        self.nz = nz # Size of z latent vector (i.e. size of generator input)
        
        # save file locations
        self._checkDirectory(save_dir) # check whether save dir exists
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        self._checkDirectory(self.checkpoint_dir) # create checkpoints dir
        self.fake_image_dir = os.path.join(save_dir, 'fake_images')
        self._checkDirectory(self.fake_image_dir) # create fake images dir
        
        # model parameters
        self.num_epochs = num_epochs # number of epochs to train for
        self.start_epoch = 1 # the starting epoch
        self.criterion = criterion # loss function
        self.real_label = 1 # Establish convention for real and fake labels during training
        self.fake_label = 0

        # networks init
        self.netD = netD
        self.netG = netG
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        
        # device
        self.device = device # specify device being used
        
        # Create fixed noise to visualize the progression of the generator
        self.fixed_noise = torch.randn(64, self.nz, device=self.device) # torch.Size([64, 100])
            
        
    def generate_fake_images(self, class_index_tensor, noise, image_name = 'random',
                             save = True, ncols = 8):
        
        '''
        Generate a batch of fake images using current generator weights.
        
        Inputs
        
            class_index_tensor (LongTensor)
                The class index to create fake images for. The number of fake images generated is equal
                to the length of the tensor. So a tensor filled with 10 "1"s will generate 10 images for
                the class that corresponds to "1".
                
            noise (Tensor)
                Random noise that will be put through the generator weights to produce an image.
        
            image_name (STR)
                Image name for the saved file.
                If running this function in model training, image_name should contain a changing variable,
                otherwise the files will just keep overwriting each other with the same name.
                Default: 'random' (in case save = True but no image_name provided)
            
            save (BOOL)
                If save is TRUE, the image file will be saved in the specified "self.fake_image_dir".
                Otherwise, just return the image data for plotting.
                Default: TRUE
            
        ''' 
        with torch.no_grad():
            # create fake images for a the labels in class_index_tensor
            fake = self.netG(noise, class_index_tensor).detach().cpu()
        
        if save: # save images in the fake_image_dir
            save_image(fake.data, f'{self.fake_image_dir}/{image_name}.png',
                       nrow=ncols, padding=2, normalize=True)
        
        return fake.data
    

    def train(self):
        
        '''
        Training loop
        '''
        if self.num_epochs == 0:
            print(f"No epochs set for training. Exiting training loop.")
            return
            
        # Lists to keep track of progress
        self.G_losses = [] # generator loss
        self.D_losses = [] # discriminator loss
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            # For each batch in the dataloader
            for i, (imgs, class_labels) in enumerate(self.dataloader):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                ## Train with all-real batch
                ###########################
                self.netD.zero_grad()

                # Format batch
                real_imgs = imgs.to(self.device)
                b_size = real_imgs.size(0)

                # Set ground truth labels as REAL
                validity_label = torch.full((b_size,), self.real_label, device=device)

                # Forward pass real batch through D
                output = self.netD(real_imgs, class_labels).view(-1)

                # Calculate loss on all-real batch
                errD_real = self.criterion(output, validity_label)

                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()


                ## Train with all-fake batch
                ###########################
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, device=device) # torch.Size([128, 10])

                # Generate batch of fake labels
                gen_labels = torch.randint(self.n_classes, (b_size,)).type(torch.LongTensor) # torch.Size([128, 3])

                # Generate fake image batch with G
                fake = self.netG(noise, gen_labels)

                # Update ground truth labels to FAKE
                validity_label.fill_(self.fake_label)

                # Classify all fake batch with D
                output = self.netD(fake.detach(), gen_labels).view(-1)

                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, validity_label)

                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake

                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                self.netG.zero_grad()

                validity_label.fill_(self.real_label)  # fake labels are real for generator cost
                
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake, gen_labels).view(-1)
                
                # Calculate G's loss based on this output
                errG = self.criterion(output, validity_label)
                
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print(f'[{epoch}/{self.start_epoch + self.num_epochs - 1}][{i}/{len(self.dataloader)}]\tLoss_D: {round(errD.item(),2)}\tLoss_G: {round(errG.item(),2)}\tD(x): {round(D_x,2)}\tD(G(z)): {round(D_G_z1/D_G_z2,2)}')

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                # every 500 iterations, or on the last batch of the last epoch
                if (iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i == len(self.dataloader)-1)):
                    
                    print("Saving a batch of fake images.")
                    
                    class_index = torch.arange(self.n_classes) # get class indices
                    for i in class_index:
                        class_index_tensor = torch.LongTensor(64).fill_(i) # repeat the same class index 10 times
                        self.generate_fake_images(class_index_tensor, self.fixed_noise,
                                                  image_name = f'{self.classes[i]}_e{epoch}', save = True)

                iters += 1

            # automatically save model for first epoch (testing) and every 5 epochs
            if epoch == 1 or epoch % 5 == 0: self.save(epoch)

        print(f"Finished Training for {epoch} epochs.")
        self.save(epoch)
        
        
    def save(self, epoch):
        
        # save the model checkpoint
        filepath = f'{self.checkpoint_dir}/checkpoint_e{epoch}.pth.tar'
        print(f"=> Saving checkpoint: {filepath}")

        state = {
            'D_losses': self.D_losses,
            'G_losses': self.G_losses,
            'epoch': epoch,
            'netD_state_dict': self.netD.state_dict(),
            'optimizerD': self.optimizerD.state_dict(),
            'netG_state_dict': self.netG.state_dict(),
            'optimizerG': self.optimizerG.state_dict(),
        }

        torch.save(state, filepath) 

        
    def load(self, loadpath, disp = False):
        '''
        When loading model checkpoint, just load the epoch and state dicts to continue training.
        The D-loss and G-loss can be stored within their respective checkpoints
        and referred to later when needed.
        '''
        if os.path.isfile(loadpath):
            
            if disp: print(f"=> loading checkpoint: {loadpath}")

            checkpoint = torch.load(loadpath)

            self.start_epoch = checkpoint['epoch'] + 1
            self.netD.load_state_dict(checkpoint['netD_state_dict'])
            self.netG.load_state_dict(checkpoint['netG_state_dict'])
            self.optimizerD.load_state_dict(checkpoint['optimizerD'])
            self.optimizerG.load_state_dict(checkpoint['optimizerG'])

            if disp:
                print(f"=> loaded checkpoint: {loadpath}")
                print(f"Last epoch was {checkpoint['epoch']}")

        else: 
            print(f"=> No checkpoint found at: {loadpath}")

        