/*
 OCRC, a AI for optical character recognition written in C
 Copyright (C) 2023-2023 João Edison Roso Manica

 OCRC is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 OCRC is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <png.h>
/*1.6.37*/
#include <stdlib.h>
#include <cblas.h>
#include <math.h>

#define WIDTH 128
#define HEIGHT 128
#define POOL_LEN 8
#define DIM_POOL (WIDTH / POOL_LEN)
#define PIXEL_QTT (DIM_POOL * DIM_POOL)
#define DIM_IMG1 (DIM_POOL - 2)
#define DIM_IMGL (DIM_IMG1)
#define AREA_IMG (DIM_IMGL * DIM_IMGL)
#define METADATA_QTT 28
#define FEATURE_QTT 1
#define INPUT_QTT (FEATURE_QTT * AREA_IMG + METADATA_QTT)
#define MAX_CLASSES 36

/*Activation function used in all layers except in the last one*/
#define ACTIVATION_FN(X) tanh(X)
#define DERIVATIVE_ACTIVATION_FN(Z) (1 - powf(tanh(Z), 2))
/*
#define ACTIVATION_FN(X) (1 / (1 + exp(-X)))
#define DERIVATIVE_ACTIVATION_FN(Z) (ACTIVATION_FN(Z) * (1 - ACTIVATION_FN(Z))) 
*/

/*Learning rate and momentum*/
#define RATE 0.5
#define MOMENTUM 0.3

/* 'neurons_per_layer': This vector defines the number of neurons at each layer. 
   (Feedforward sense: first to last)
 * 'num_layers': Indicates the total number of layers in the net
 * 'num_input': Represents the total number of inputs in the network
 * 'source': Specifies whether the net receives inputs from convolution
 * 'output': Specifies the index of the net that receives the output of the current net as input.
   (-1 indicates that the output is the last layer of the network)
 * */
struct create_network {
	unsigned *neurons_per_layer, num_layers, num_input;
	unsigned char source;
	short output;
};

extern float *network_output;

/* read_png_file:
 * 	Reads 'name' with fopen and verifies if it is an png file.
 * 	Returns 0 if succeeded correctly and 'img_view' gets the output.
 * 	Else returns a value of the error and prints a message.
 * 	Is apllied a convolution and metadata is extracted.
 * 	'verbose' shows png characteristics of the file. 
 * 	'img_view' needs to be INPUT_QTT len.
 * */
int read_png_file(char name[], float *img_view, int verbose);

/* hit:
 * 	Check if the greatest propability predicted by the net is equals to expected 'class'.
 * 	Returns 1 if true and 0 if false. This needs to be float to use this function in 
  	pointers that uses 'cross_entropy' too.
 * 	'predi' gets the class index predicted by the net and 'predv' gets the probability. 
 * */
float hit(int class, int *predi, float *predv);

/* cross_entropy:
 * 	Computes the Cross Entropy in nats of the net output.
 * 	'class' is the index of the expected class.
 * */
float cross_entropy(int class);

/* init_net_topology:
 * 	Use 'nets' array to assemble the network.
 * 	'n' is the number of nets in the network.
 * 	'verbose' shows messages that says that this begins and ends.
 * */
void init_net_topology(struct create_network nets[], int n, int verbose);

/* init_random_weights:
 * 	All the bias and weights of the network gets random values.
 * 	The net needs to be loaded in memory by 'load_weights' or 'init_net_topology'.
 * */
void init_random_weights();

/* load_weights:
 * 	reads `weights` file and loads the network.
 * 	'verbose' shows messages that inform that this begin and end.
 * */
void load_weights(int verbose);

/* save_weights:
 * 	write in `weights` file the network.
 * */
void save_weights();

/* run:
 * 	Computes feedforward using 'img_view' and 'network_output' gets the output of the network
 * */
void run(float *img_view);

/* ini_backpr:
 * 	Allocates memory necessary to make the backpropagation.
 * 	'n' is used to calculate the mean of the samples in one batch.
  	If this is 1 then no mean is calculated.
 * */
void ini_backpr(int n);

/* clear_backpr:
 *	Clean the values stored in a backpropagation iteration.		
 * */
void clear_backpr();

/* backpr:
 *	Computes the backpropagation using the sample 'img_view'. It needs 'run' before!
 *	'expected' is the index of the expected class output.
 * */
void backpr(float *img_view, int expected);

/* apply_backpr:
 * 	Modify the weights and biases using the gradients of error get by 'backpr' and momentum
 * */
void apply_backpr();

/* end_backpr:
 * 	Deallocates memory of the backpropagation.
 * */
void end_backpr();
