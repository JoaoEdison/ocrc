/*
 OCRC, a AI for optical character recognition writed in C
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

unsigned layers1[] = {64, 32, 32, 24};
unsigned layers2[] = {28, 12, 12, 12};
unsigned layers3[] = {36, 36, 36, MAX_CLASSES};

struct create_network nets[] = {
	{layers1, sizeof(layers1)/sizeof(unsigned), DIM_IMG * DIM_IMG, 1,  2},
	{layers2, sizeof(layers2)/sizeof(unsigned),      METADATA_QTT, 1,  2},
	{layers3, sizeof(layers3)/sizeof(unsigned),                36, 0, -1}
};

main()
{
	init_net_topology(nets, sizeof(nets)/sizeof(struct create_network), 1);
	init_random_weights();
	save_weights();
}
