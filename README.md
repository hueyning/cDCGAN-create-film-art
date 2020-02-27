# Conditional DCGAN (cDCGAN) for Generating Artwork Conditioned on Film Genres

This repo contains a cDCGAN model that I trained to generate artwork based on different film genres, e.g. cartoon, cyberpunk, horror, noir, western. The cDCGAN model was modified based on the official Pytorch tutorial (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) and the conditionals (the film genres) were embedded as a final linear layer to the cDCGAN network.

I trained the model on both the CIFAR10 dataset as well as my own novel dataset of digital art scraped from https://www.artstation.com/ using the web-scraper located in https://github.com/hueyning/art-station-scraper.

The results of the model can be seen in `EvaluateGan-ArtDataset.ipynb`.

### Repository Structure

```
Jupyter Notebooks
├── EvaluateGan-ArtDataset.ipynb        # notebook for cDCGAN trained on ArtStation dataset
├── EvaluateGan-CIFARDataset.ipynb      # notebook for cDCGAN trained on CIFAR dataset
├── FID-ArtDataset.ipynb                # FID calculation between generated vs real ArtStation dataset
├── FID-CIFARDataset.ipynb              # FID calculation between generated vs real CIFAR10 dataset
├── VisualizeImages-FullRes.ipynb       # Full-res visualization of real ArtStation dataset

Python Files
├── modules.py                          # contains all imported libraries and packages
├── cdcgan_model
|   ├── cdcgan.py                       # cdcgan class
|   ├── generator_discriminator_32.py   # generator and discriminator network classes for images of size 32 x 32 x 3
|   ├── generator_discriminator_64.py   # generator and discriminator network classes for images of size 64 x 64 x 3
|   └── weights_init.py                 # function to initialize weights
├── eval_metrics
|   ├── FID.py                          # functions for calculating FID
|   ├── inception_network.py            # InceptionV3 architecture (used in FID.py)
|   └── inception_score.py              # functions for calculating IS
└── helper_funcs
    ├── generate_images.py              # contains functions to generate images using the model
    └── plot_and_index.py               # contains helper functions for visualization & indexing
```

### References
- The cDCGAN model was modifed from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- The Inception Score was modified from: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
- The FID was modified from: https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
