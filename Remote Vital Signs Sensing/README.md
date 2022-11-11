# Contactless Vital Signs Sensing code examples
This folder contains the code example for my work as an undergraduate research assistant for the MTTS-CAN research group. My main work was to create different variations of video data using background and lighting augmentations, then evaluate a deep learning model on the new data to determine the effect of data augmentation on model performance.

## Main codes

**examine_data.py** -- Reads a .mat file, shows the first image in the video sequence, and prints the shape as well as a sample of the vital sign data

**augmentations.py** -- The main file used for background and lighting augmentations. There are several methods: `bg_aug_simple` augments background by thresholding pixel values, `bg_aug_matte` uses alpha mattes produced with [MODNET](https://github.com/ZHKKKe/MODNet) to augment the video background. `lighting_aug_gamma` and `lighting_aug_alpha` does gamma lighting augmentation and alpha-beta lighting augmentation, respectively.

**plot.py** -- Plots an image grid containing a snapshot of the video, its normalized version, the corresponding attention mask, as well as the output of the model, which is a waveform representing the heartbeats.

**vid2mat.py** -- Data preprocessing function that turns videos into .mat files, which is used as the input for the model.

## The following codes are used for data organization

**chunk.py**

**combine_chunks.py**

**data_file_organizer.py**
