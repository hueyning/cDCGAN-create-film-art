# Conditional DCGAN (cDCGAN) for Generating Artwork Conditioned on Film Genres

This repo contains a cDCGAN model that I am training to generated artwork based on different film genres, e.g. noir, horror, cyberpunk, western, cartoon. The cDCGAN model was built based on the DCGAN model tutorial written in Pytorch (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) and the conditionals (the film genres) were embedded using a `nn.embedding` layer and attaching it to the final linear layer of the cDCGAN network.

I tested the model on both the CIFAR10 dataset as well as my own novel dataset of digital art collected from https://www.artstation.com/ so that I could test the model for bugs during the early stages, and compare the final output.

This is still a W.I.P. but should be completed soon, at which point I will upload the final paper on this repo as well.

```
root
├── EvaluateGan-ArtDataset.ipynb # main notebook for implementing the model
├── modules.py # contains all imported libraries and packages
├── cdcgan_model
|   ├── cdcgan.py # cdcgan class
|   ├── generator_discriminator_32.py # generator and discriminator network classes for images of size 32 x 32 x 3, i.e. the CIFAR10 dataset
|   ├── generator_discriminator_64.py # generator and discriminator network classes for images of size 64 x 64 x 3, i.e. the novel dataset
|   └── weights_init.py # function to initialize weights
├── eval_metrics
|   ├── FID.py # functions for calculating FID
|   ├── inception_network.py # InceptionV3 architecture (used in FID.py)
|   └── inception_score.py # functions for calculating IS
└── helper_funcs
    ├── generate_images.py # contains functions to generate images using the model
    └── plot_and_index.py # contains misc functions for visualization & indexing
```
