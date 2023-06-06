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
#define END_LAYER_2 METADATA_QTT
#define CONNECTION (END_LAYER_1 + END_LAYER_2)

unsigned layers1[] = {END_LAYER_1};
unsigned layers2[] = {END_LAYER_2};
unsigned layers3[] = {CONNECTION, CONNECTION, 64, 64, 36, MAX_CLASSES};

/*
 * `create_network` array must adhere to the following rules:
 *   1) If multiple nets receive inputs from convolution, the nets listed first in the array receive inputs in the same order.
 *   2) Only one net can serve as the final output (-1) of the network.
 *   3) To specify that a net receives inputs from other nets: 
 *   	  In the `output` of each net that sends inputs, you must indicate the index of the receiving net.
 *	  The receiving net must set its `num_input` to the sum of the lats layers of the source nets.
 * */
struct create_network nets[] = {
	{layers1, sizeof(layers1)/sizeof(unsigned), FEATURE_QTT * AREA_IMG, 1,  2},
	{layers2, sizeof(layers2)/sizeof(unsigned),           METADATA_QTT, 1,  2},
	{layers3, sizeof(layers3)/sizeof(unsigned),             CONNECTION, 0, -1}
};

main()
{
	init_net_topology(nets, sizeof(nets)/sizeof(struct create_network), 1);
	init_random_weights();
	save_weights();
}
