# function for generating fake images using a trained GAN model

import uuid

def _checkDirectory(dirName):

    if not os.path.exists(dirName):
        print(f"{dirName} directory does not exist. Making {dirName}")
        os.makedirs(dirName)

    else: print(f"{dirName} directory exists.")


def generate_fake_images(model, classes, number_of_images, nz, 
                         device, save_dir, save = True):
    
    '''
    Generates a batch of fake images (len(classes) x number_of_images)
    using current generator weights.

    Inputs

        model (OBJ)
            The Pytorch model to be used for generating the fake images.

        classes (ARR)
            List of class labels, where the position in the list corresponds to the class index.

        number_of_images (INT)
            The number of fake images to generate.
            
        nz (INT)
            Size of z latent vector.
            
        save_dir (STR)
            The directory in which to save the generated images.

        save (BOOL)
            If save is TRUE, the image file will be saved in a folder with the class name.
            Otherwise, just return the image data for plotting.
            Default: TRUE
    ''' 
    
    if number_of_images == 0:
        print("No images to generate. Exiting function.")

    else:
        for i in range(len(classes)):
            
            # check that the save directory exists. Else, create it.
            folder_name = classes[i]
            dir_name = os.path.join(save_dir, folder_name)
            _checkDirectory(dir_name)
            
            print(f"Generating {number_of_images} images for {folder_name}")
            
            class_index_tensor = torch.LongTensor(number_of_images).fill_(i)
            noise = torch.randn(number_of_images, nz, device=device)
            
            with torch.no_grad():
                # create fake images for a the labels in class_index_tensor
                fake = model.netG(noise, class_index_tensor).detach().cpu()

                if save: # save images in the dir_name
                    for i in range(len(fake.data)):
                        
                        # generate random file name using uuid hash
                        file_name = f'{dir_name}/{uuid.uuid4().hex[:10]}.png'
                        
                        # if hash collision, generate another hash
                        while os.path.exists(file_name):
                            file_name = f'{dir_name}/{uuid.uuid4().hex[:10]}.png'
                            
                        save_image(fake.data[i], file_name, normalize=True)
                
                