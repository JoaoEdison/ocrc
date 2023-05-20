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

* *init_net.out*, the program that generate the raw neural network.

* *training.out*, the training program.

* *view.out*, one program to see the results.

These programs look for the `weights` file in the current working directory to save and load the neural network.

## License

This project is licensed under [GPL-3.0](https://raw.githubusercontent.com/Illumina/licenses/master/gpl-3.0.txt). 
