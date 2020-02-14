# Conditional DCGAN (cDCGAN) for Generating Artwork Conditioned on Film Genres

This repo contains a cDCGAN model that I am training to generated artwork based on different film genres, e.g. noir, horror, cyberpunk, western, cartoon. The cDCGAN model was built based on the DCGAN model tutorial written in Pytorch (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) and the conditionals (the film genres) were embedded using a `nn.embedding` layer and attaching it to the final linear layer of the cDCGAN network.

I tested the model on both the CIFAR10 dataset as well as my own novel dataset of digital art collected from https://www.artstation.com/ so that I could test the model for bugs during the early stages, and compare the final output.

This is still a W.I.P. but should be completed soon, at which point I will upload the final paper on this repo as well.
