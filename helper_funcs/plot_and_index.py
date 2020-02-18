# functions for plotting images and setting indices in dataset

# import necessary libraries
from modules import *
from torch.utils.data import SubsetRandomSampler

def plot_images(dataloader, classes, device, image_number = 8, disp_labels = True, model = None):
    
    '''
    Function to plot a sample of images from the dataloader, alongside their class labels.
    If a model is assigned to the model parameter, the predicted labels will be printed as well.
    
    Input:
        dataloader (DATALOADER)
            Dataloader of dataset.
            
        classes (ARR)
            Array type object containing the class labels (strings) in the order that 
            corresponds with the numerical key in the dataloader.
        
        image_number (INT)
            Number of images to plot from the dataloader. image_number should not exceed batch size.
            Since images are plotted in a row, any number > 10 could cause display issues.
            Default: 8.

        disp_labels (BOOL)
            If True, then True labels will be displayed. Works best with image_number < 8 
            since the True labels should be displayed above the images.
            If False, labels will not be displayed. Ideal for plotting large batch of images.
            Default: True
        
        model (PYTORCH MODEL)
            Optional parameter. If a model is provided, the predicted labels from the 
            model for each of the images will be printed as well. 
            Default: None.
    '''
    
    # get images and true labels
    images, labels = next(iter(dataloader))

    # plot images
    plt.figure(figsize=(16,16))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(images.to(device)[:image_number], padding=1, normalize=True).cpu(),(1,2,0)))
    
    # print true labels
    if disp_labels:
        print('True labels: ', '     '.join('%5s' % classes[labels[j]] for j in range(image_number)))
    
    if model:
        # predict image classes using custom net
        outputs = model(images)
        # the outputs are energies for the 10 classes. 
        # the higher the energy for a class, the more the network thinks that the image is of the particular class.
        # So, we get the index of the highest energy:
        _, predicted = torch.max(outputs, 1)
        # print predicted labels
        print('Predicted:  ', '   '.join('%5s' % classes[predicted[j]] for j in range(image_number)))


def get_target_index(dataset):
    '''
    Given a dataset, this function returns a dictionary of classes, where the value of each class 
    is a dictionary containing the class indices and the number of datapoints in the class.
    
    Input:
        dataset (IMAGEFOLDER)
            Dataset should be ImageFolder class.
        
    Output:
        idx_dct (DCT)
            Nested dictionary with the class name as key, and a dictionary containing the
            'indices' and 'length' of the class as values.
            Example format:
            idx_dct = { 'class_A':{
                        'indices': [1,2,3,4,5],
                        'length': 5
                        },
                        'class_B':{
                        'indices': [6,7,8],
                        'length': 3
                        },
                        'class_C':{
                        'indices': [100,101,102,103],
                        'length': 4
                        }}
    '''
    targets = torch.tensor([t[1] for t in dataset.samples])
    idx_dct = {}
    
    for k,v in dataset.class_to_idx.items():
        idx_dct[k] = {'indices': (targets == v).nonzero().reshape(-1)}
        idx_dct[k]['length'] = len(idx_dct[k]['indices'])
        
    return idx_dct


def plot_gen_images(classes, model, nz, device, no_of_images = 10):
    '''
    Plots the generated images for each class.
    
    Inputs
        classes (ARR)
            List of classes to be used as labels for generating fake images.
        
        model (MODEL)
            Pytorch model to be used for generating fake images.
        
        no_of_images (INT)
            Number of fake images to generated for each class.
            Default: 10.
    '''
    noise = torch.randn(no_of_images, nz, device=device)

    for c in range(len(classes)):
        images = model.generate_fake_images(torch.LongTensor(no_of_images).fill_(c), noise = noise, save = False)
        plt.figure(figsize=(20,10))
        plt.axis('off')
        plt.title(f'Fake {classes[c]}', size=20)
        plt.imshow(np.transpose(vutils.make_grid(images, nrow = no_of_images, normalize = True), (1,2,0)))


def plot_progression(model, classes, class_name, checkpoint_dir, epoch_list, no_of_images, nz, device):
    '''
    Plot the progression of image generation by generating images
    based on different training epochs of the model.
    
    '''
    noise = torch.randn(no_of_images, nz, device=device)
    
    for epoch in epoch_list:
        
        # load model weights for a given epoch
        curr_model = model.load(f'{checkpoint_dir}/checkpoint_e{epoch}.pth.tar')
        
        class_index = classes.index(class_name)
        images = model.generate_fake_images(torch.LongTensor(no_of_images).fill_(class_index), noise = noise, save = False)
        
        plt.figure(figsize=(20,10))
        plt.axis('off')
        plt.title(f'Fake {class_name} (epoch = {epoch})', size=20)
        plt.imshow(np.transpose(vutils.make_grid(images, nrow = no_of_images, normalize = True), (1,2,0)))


def plot_batch(dataset, class_name, batch_size = 25, nrow = 5, device = 'cpu'):    
    '''
    Plot images from a set based on a given class name
    '''
    class_index = dataset.classes.index(class_name)
    target_idx_dct = get_target_index(dataset)
    dataloader = torch.utils.data.DataLoader(
                                    dataset, batch_size = batch_size, 
                                    sampler = SubsetRandomSampler(
                                    target_idx_dct[dataset.classes[class_index]]['indices']))
    batch = next(iter(dataloader))
    plt.figure(figsize=(20,12))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(batch[0].to(device)[:100],
                padding=2, nrow=nrow, normalize=True).cpu(),(1,2,0)))
    

