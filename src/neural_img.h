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

#define WIDTH 128
#define HEIGHT 128
#define POOL_LEN 8
#define DIM_POOL (WIDTH / POOL_LEN)
#define PIXEL_QTT (DIM_POOL * DIM_POOL)
#define DIM_IMG1 (DIM_POOL - 2)
#define DIM_IMGL (DIM_IMG1)
#define AREA_IMG (DIM_IMGL * DIM_IMGL)
#define METADATA_QTT (28 + DIM_POOL * 2)
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
#define RATE 1.0f
#define MOMENTUM 0.3f

/* 'neurons_per_layer': This vector defines the number of neurons at each layer. 
   (Feedforward sense: first to last)
 * 'num_layers': Indicates the total number of layers in the net
 * 'num_input': Represents the total number of inputs in the network
 * 'source': Specifies whether the net receives inputs from convolution
 * 'output': Specifies the index of the net that receives the output of the current net as input.
   (-1 indicates that the output is the last layer of the network)
 * */
typedef struct {
	unsigned *neurons_per_layer, num_layers, num_input;
	unsigned char source;
	short output;
} create_network, create_network_arr[], *create_network_ptr;

typedef struct {
	struct net *arr;
	unsigned num_classes;
	unsigned char back_on, num_nets;
	float N;
	float *network_output;
} bignet, *bignet_ptr;

/* read_png_file:
 * 	Reads 'name' using fopen and verifies if it is a PNG file.
 * 	The function applies convolution to the image and extracts metadata.
 * 	Returns 0 if the file is successfully read and the output is stored in 'img_view'.
 * 	Else returns an error value and prints an error message.
 * 	If 'verbose' is not zero, it displays the characteristics of the file. 
 * 	'img_view' should have a length of INPUT_QTT.
 * */
int read_png_file(char name[], float *img_view, int verbose);

/* hit:
 * 	Checks if the highest probability predicted by the net is equal to the expected 'class'.
 * 	Returns 1 if true and 0 if false. This needs to be float to enable 
 	using with pointers that also use 'cross_entropy'.
 * 	'predi' receives the class index predicted by the net.
 * 	'predv' receives the corresponding probability.
 * */
float hit(bignet_ptr model, int classidx, int *predi, float *predv);

/* cross_entropy:
 * 	Calculates the Cross Entropy, measured in nats, of the net output.
 * 	'class' is the index of the expected class.
 * */
float cross_entropy(bignet_ptr model, int classidx);

/* init_net_topology:
 * 	Uses 'nets' array to assemble the network.
 * 	'n' is the number of nets in the network.
 * 	If 'verbose' is true, it displays messages indicating the start and end.	
 * */
bignet_ptr init_net_topology(create_network_arr nets, int n, int verbose);

/* load_weights:
 * 	Reads 'file_name' file and loads the network.
 * 	If 'verbose', displays messages indicating the start and end.
 * */
bignet_ptr load_weights(char file_name[], int verbose);

/* init_random_weights:
 * 	Assigns random values to biases and weights of the network.
 * 	Is assumed that net has already been loaded into memory using 'load_weights' or
  	'init_net_topology' functions.
 * */
void init_random_weights(bignet_ptr model);

/* save_weights:
 * 	Writes in file_name the network.
 * */
void save_weights(bignet_ptr model, char file_name[]);

/* run:
 * 	Computes feedforward using 'flat_input' and, in the model, 'network_output' receives the output of the network.
 * */
void run(bignet_ptr model, float *flat_input);

/* ini_backpr:
 * 	Allocates the necessary memory for backpropagation.
 * 	'n' is used to calculate the mean of the samples in one batch.
  	If this is equal to 1, no mean CAN be calculated.
 * */
void ini_backpr(bignet_ptr model, int n);

/* clear_backpr:
 *	Clears the values stored during a backpropagation iteration.		
 * */
void clear_backpr(bignet_ptr model);

/* backpr:
 *	Performs the backpropagation using the sample 'flat_input'.
 *	'expected' is the expected class array.
 * */
void backpr(bignet_ptr model, float *flat_input, float *expected);

/* apply_backpr:
 * 	Modifies the weights and biases using the gradients of error obtained from 'backpr' and applies momentum.
 * */
void apply_backpr(bignet_ptr model);

/* end_backpr:
 * 	Deallocates memory used during the backpropagation.
 * */
void end_backpr(bignet_ptr model);
