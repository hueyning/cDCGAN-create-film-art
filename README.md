# Conditional DCGAN (cDCGAN) for Generating Artwork Conditioned on Film Genres

This repo contains a cDCGAN model that I am training to generated artwork based on different film genres, e.g. noir, horror, cyberpunk, western, cartoon. The cDCGAN model was built based on the DCGAN model tutorial written in Pytorch (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) and the conditionals (the film genres) were embedded using a `nn.embedding` layer and attaching it to the final linear layer of the cDCGAN network.

I tested the model on both the CIFAR10 dataset as well as my own novel dataset of digital art collected from https://www.artstation.com/ so that I could test the model for bugs during the early stages, and compare the final output.

This is still a W.I.P. but should be completed soon, at which point I will upload the final paper on this repo as well.

At this point, the notebooks contain the following functions:
- visualize-images.ipynb: contains functions to load images from a folder and display them in the notebook. Also contains code to evaluate the Inception Score (IS) of a dataset. 
- generate-fake-images-for-evaluation.ipynb: contains code for my cDCGAN model and functions to generate new images from the latest training checkpoint of the model.
- capstone-model.ipynb: contains the cDCGAN model code. Save directories can be named in the kwargs 'save_dir' when intializing the model object. Training checkpoints are saved at every 5 epochs in 'save_dir/checkpoints'; a batch of fake images are generated each epoch and saved in 'save_dir/fake_images'.
