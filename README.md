<div align="center">

<h1>OCRC</h1>

AI for <b>O</b>ptical <b>C</b>haracter <b>R</b>ecognition (<b>OCR</b>) written in <b>C</b>.

</div>

## Overview

Is a program that trains an artificial neural network based on a set of images, subsequently recognizes new images provided to the network.

It is only possible to read 128x128 png images, grayscale or pallete types. Until now, it only runs on Linux.

The `weights` file contains the weights and biases of the network, which have already been trained for recognition on [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) EnglishFnt dataset, more specifically on the first half of the first 36 samples.

## Requirements

* [libpng](http://www.libpng.org) >= 1.6.37

* CBLAS

## Compiling

Run makefile in `src` folder. You will get three executable codes:

* `init_net.out`, the program that generate the raw neural network.

* `training.out`, the training program.

* `view.out`, a program to see the results.

These programs look for the `weights` file, created by `init_net.out`, in the current working directory to save and load the neural network.

## Usage

You can configure the neural network format in the `init_net.c` and recompile it with the make command. Afterwards, `init_net.out` is called to generate the network, in the `weights` file. Or use the already trained `weights` file that comes in with it. (WARNING: this program overwrites `weights` on every call)

`training.out` has several training options, feel free to try them out. It is necessary to inform the absolute path for the training and test directories, respectively, in the `dataset_paths` and `test_paths` files that you need to create, they need read permissions. One absolute path must be informed per line, each of which corresponds to the class of its line in the file.

The `view.out` program is used to check the recognitions made by the network and also has several options.

## License

This project is licensed under [GPL-3.0](https://raw.githubusercontent.com/Illumina/licenses/master/gpl-3.0.txt). 
