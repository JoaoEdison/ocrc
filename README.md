<div align="center">

<h1>OCRC</h1>

AI for <b>O</b>ptical <b>C</b>haracter <b>R</b>ecognition (<b>OCR</b>) written in <b>C</b>.

<img src="https://github.com/JoaoEdison/ocrc/blob/main/architecture.png">

</div>

## Overview

A program that trains an artificial neural network based on a set of images and subsequently recognizes new images provided to the network.

It is only possible to read grayscale or pallete type PNG images with dimensions of 128x128 pixels. Until now, it only runs on Linux.

The `weights` file contains pre-trained weights and biases of the neural network. These weights were trained for recognition using the [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) EnglishFnt dataset, specifically on the first half of the first 36 samples.

## Requirements

* [libpng](http://www.libpng.org) >= 1.6.37

## Compiling

Run makefile in `src` folder. You will get three executables:

* `init_net`: this generates the raw neural network.

* `training`: the training program.

* `view`: a program to view the results.

These programs look for the `weights` file in the current working directory to save and load the neural network.

## Usage

To configure the neural network format, modify `model.h`, `model.c` or `init_net.c` and recompile it using the make command. After recompiling, execute `init_net` to generate the raw network, wich will be saved in `weights` (WARNING: running this program will overwrites `weights` on every call). Alternatively, you can use the provided pre-trained `weights`.

`training` offers several training options. Please feel free to exlore and experiment with them. To use the program, create two files where it executes: `dataset_paths` and `test_paths`. In the `dataset_paths` file, specify the absolute paths to the training directories, one path per line, corresponding to each class. Do the same for `test_paths`, where paths are to test directories. Ensure that these files have read permissions.

The `view` program allows you to inspect the network's recognition results. It also offers various options for visualization.

The file `weights` was obtained with command:
```console
./training 505 -metric=c -method=a
```

### Quick Test

If you use Ubuntu, try:

```console
sudo apt install libpng-dev
git clone https://github.com/JoaoEdison/OCRC.git
cd OCRC/src/
make
./build/view -d ../test
```

## License

This project is licensed under [GPL-3.0](https://raw.githubusercontent.com/JoaoEdison/ocrc/main/LICENSE). 
