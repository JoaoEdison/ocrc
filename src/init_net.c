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

#include "neural_img.h"

#define END_LAYER_1 64
#define END_LAYER_2 28
#define CONNECTION (END_LAYER_1 + END_LAYER_2)

unsigned layers1[] = {END_LAYER_1};
unsigned layers2[] = {END_LAYER_2};
unsigned layers3[] = {64, MAX_CLASSES};

/* 'create_network' array must adhere to the following rules:
 *   1) The order of nets in the array corresponds to the order in which they receive inputs from convolution or from other net.
     'num_input' determines the number of inputs that net receives.
 *   2) Only one net can serve as the final output (-1) of the network.
 *   3) To specify that a net receives inputs from other nets: 
 *   	  In the 'output' of each net that sends inputs, you must indicate the index of the receiving net.
 *	  The receiving net must set its 'num_input' to the sum of the lasts layers of the source nets.
 * */
create_network_arr nets = {
	{layers1, sizeof(layers1)/sizeof(unsigned), FEATURE_QTT * AREA_IMG + DIM_POOL * 2, 1,  2},
	{layers2, sizeof(layers2)/sizeof(unsigned),                                    28, 1,  2},
	{layers3, sizeof(layers3)/sizeof(unsigned),                            CONNECTION, 0, -1}
};

main()
{
	bignet_ptr model;

	model = init_net_topology(nets, sizeof(nets)/sizeof(create_network), 1);
	init_random_weights(model);
	save_weights(model, "weights");
}
