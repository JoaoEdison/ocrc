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

struct create_network {
	unsigned *neurons_per_layer, num_layers, num_input;
	unsigned char source;
	short output;
};

extern float *network_output;

int read_png_file(char[], float *, int);
float hit(int, int*, float*);
float cross_entropy(int);
void init_net_topology(struct create_network[], int, int);
void init_random_weights();
void load_weights(int);
void save_weights();
void run(float*);
void ini_backpr(int);
void clear_backpr();
void backpr(float*, int);
void apply_backpr();
void end_backpr();
