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

struct layer {
	float **w, *b, *z, *a;
	float **err_w, *err_b, *aux_b;
	float **change_w, *change_b; 
	unsigned n, prev_n;
};

struct net {
	struct layer *arr;
	float *input_first;
	struct net **in_nets;
	unsigned char num_layers, output_original, num_in_nets;
	short out_id;
};

struct {
	struct net *arr;
	float N;
	unsigned num_classes;
	unsigned char back_on, num_nets;
} bignet;


#define ACTIVATION_FN(X) tanh(X)
#define DERIVATIVE_ACTIVATION_FN(Z) (1 - powf(tanh(Z), 2))

/*
#define ACTIVATION_FN(X) (1 / (1 + exp(-X)))
#define DERIVATIVE_ACTIVATION_FN(Z) (ACTIVATION_FN(Z) * (1 - ACTIVATION_FN(Z))) 
*/

void run(img_view)
float *img_view;
{
	struct net *ptrn;
	struct layer *ptrl;
	float sum;
	int i;

	for (ptrn=bignet.arr; ptrn < bignet.arr + bignet.num_nets; ptrn++) {
		ptrl = ptrn->arr;
		if (ptrn->input_first)
			cblas_sgemv(CblasRowMajor, CblasNoTrans, ptrl->n, ptrl->prev_n, 1, ptrl->w[0], ptrl->prev_n, ptrn->input_first, 1, 0, ptrl->z, 1);
		else {
			cblas_sgemv(CblasRowMajor, CblasNoTrans, ptrl->n, ptrl->prev_n, 1, ptrl->w[0], ptrl->prev_n, img_view, 1, 0, ptrl->z, 1);
			img_view += ptrl->prev_n;
		}
		for (i=0; i < ptrl->n; i++) {
			ptrl->z[i] += ptrl->b[i];
			ptrl->a[i] = ACTIVATION_FN(ptrl->z[i]);
		}
		for (ptrl+=1; ptrl < ptrn->arr + ptrn->num_layers; ptrl++) {
			cblas_sgemv(CblasRowMajor, CblasNoTrans, ptrl->n, ptrl->prev_n, 1, ptrl->w[0], ptrl->prev_n, (ptrl-1)->a, 1, 0, ptrl->z, 1);
			for (i=0; i < ptrl->n; i++)
				ptrl->z[i] += ptrl->b[i];
			if (ptrl->a == network_output) {
				for (sum=i=0; i < ptrl->n; i++) {
					ptrl->a[i] = exp(ptrl->z[i]);
					sum += ptrl->a[i]; 
				}
				for (i=0; i < ptrl->n; i++)
					ptrl->a[i] /= sum;
			} else
				for (i=0; i < ptrl->n; i++)
					ptrl->a[i] = ACTIVATION_FN(ptrl->z[i]);
		}
	}
}

void ini_backpr(n)
{
	struct net *ptrn;
	struct layer *ptrl;
	int i;

	for (ptrn=bignet.arr; ptrn < bignet.arr + bignet.num_nets; ptrn++)
		for (ptrl=ptrn->arr; ptrl < ptrn->arr + ptrn->num_layers; ptrl++) {
			ptrl->err_w = malloc(sizeof(float*) * ptrl->n);
			ptrl->err_w[0] = malloc(sizeof(float) * ptrl->n * ptrl->prev_n);
			ptrl->err_b = malloc(sizeof(float) * ptrl->n);
			if (n > 1)
				ptrl->aux_b = malloc(sizeof(float) * ptrl->n);
			ptrl->change_w = malloc(sizeof(float*) * ptrl->n);
			ptrl->change_w[0] = calloc(ptrl->n * ptrl->prev_n, sizeof(float));
			ptrl->change_b = calloc(ptrl->n, sizeof(float));
			for (i=1; i < ptrl->n; i++) {
				ptrl->err_w[i] = *(ptrl->err_w) + i * ptrl->prev_n;
				ptrl->change_w[i] = *(ptrl->change_w) + i * ptrl->prev_n;
			}
		}
	bignet.N = n;
	bignet.back_on = 1;
}

void end_backpr()
{
	struct net *ptrn;
	struct layer *ptrl;

	for (ptrn=bignet.arr; ptrn < bignet.arr + bignet.num_nets; ptrn++)
		for (ptrl=ptrn->arr; ptrl < ptrn->arr + ptrn->num_layers; ptrl++) {
			free(ptrl->err_w[0]); free(ptrl->err_w); free(ptrl->err_b);
			free(ptrl->change_w[0]); free(ptrl->change_w); free(ptrl->change_b);
			if (bignet.N > 1)
				free(ptrl->aux_b);
		}
	bignet.back_on = 0;
}

void clear_backpr()
{
	struct net *ptrn;
	struct layer *ptrl;
	int i, j;
	
	for (ptrn=bignet.arr; ptrn < bignet.arr + bignet.num_nets; ptrn++)
		for (ptrl=ptrn->arr; ptrl < ptrn->arr + ptrn->num_layers; ptrl++)
			for (i=0; i < ptrl->n; i++) {
				for (j=0; j < ptrl->prev_n; j++)
					ptrl->err_w[i][j] = 0;
				ptrl->err_b[i] = 0;
			}
}

void backpr(expected, img_view)
float expected[], *img_view;
{
	struct net *ptrn, *ptrn_prev;
	struct layer *ptrl;
	int i, next_col;
	
	next_col = 0;	
	for (ptrn = bignet.num_nets - 1 + bignet.arr; ptrn >= bignet.arr; ptrn--) {
		ptrl = ptrn->num_layers - 1 + ptrn->arr;
		/*delta*/
		if (bignet.N > 1) {
			if (ptrn->out_id == -1)
				for (i=0; i < ptrl->n; i++)
					ptrl->aux_b[i] = ptrl->a[i] - expected[i];
			else
				for (i=0; i < ptrl->n; i++)
					ptrl->aux_b[i] *= DERIVATIVE_ACTIVATION_FN(ptrl->z[i]);
			for (i=0; i < ptrl->n; i++)
				ptrl->err_b[i] += ptrl->aux_b[i] / bignet.N;
			/*derivada parcial do custo para o peso*/
			do {
				cblas_sger(CblasRowMajor, ptrl->n, ptrl->prev_n, 1/bignet.N, ptrl->aux_b, 1, (ptrl-1)->a, 1, &ptrl->err_w[0][0], ptrl->prev_n);
				cblas_sgemv(CblasRowMajor, CblasTrans, ptrl->n, ptrl->prev_n, 1, &ptrl->w[0][0], ptrl->prev_n, ptrl->aux_b, 1, 0, (ptrl-1)->aux_b, 1);
				ptrl--;
				for (i=0; i < ptrl->n; i++) {
					ptrl->aux_b[i] *= DERIVATIVE_ACTIVATION_FN(ptrl->z[i]);
					ptrl->err_b[i] += ptrl->aux_b[i] / bignet.N;
				}
			} while (ptrl > ptrn->arr);
			cblas_sger(CblasRowMajor, ptrl->n, ptrl->prev_n, 1/bignet.N, ptrl->aux_b, 1, ptrn->input_first? ptrn->input_first : img_view, 1, &ptrl->err_w[0][0], ptrl->prev_n);
			if (ptrn->input_first) {
				for (ptrn_prev = ptrn->in_nets[0]; ptrn_prev < ptrn->in_nets[0] + ptrn->num_in_nets; ptrn_prev++) {
					cblas_sgemv(CblasRowMajor, CblasTrans, ptrl->n, ptrn_prev->arr[ptrn_prev->num_layers-1].n,
							1, &ptrl->w[0][next_col], ptrl->prev_n, 
							ptrl->aux_b, 1, 
							0, ptrn_prev->arr[ptrn_prev->num_layers-1].aux_b, 1);
					next_col += ptrn_prev->arr[ptrn_prev->num_layers-1].n;
				}
			} else
				img_view += ptrl->prev_n;
		} else {
			if (ptrn->out_id == -1)
				for (i=0; i < ptrl->n; i++)
					ptrl->err_b[i] = ptrl->a[i] - expected[i];
			else
				for (i=0; i < ptrl->n; i++)
					ptrl->err_b[i] *= DERIVATIVE_ACTIVATION_FN(ptrl->z[i]);
			do {
				cblas_sger(CblasRowMajor, ptrl->n, ptrl->prev_n, 1, ptrl->err_b, 1, (ptrl-1)->a, 1, &ptrl->err_w[0][0], ptrl->prev_n);
				cblas_sgemv(CblasRowMajor, CblasTrans, ptrl->n, ptrl->prev_n, 1, &ptrl->w[0][0], ptrl->prev_n, ptrl->err_b, 1, 0, (ptrl-1)->err_b, 1);
				ptrl--;
				for (i=0; i < ptrl->n; i++)
					ptrl->err_b[i] *= DERIVATIVE_ACTIVATION_FN(ptrl->z[i]);
			} while (ptrl > ptrn->arr);
			cblas_sger(CblasRowMajor, ptrl->n, ptrl->prev_n, 1, ptrl->err_b, 1, ptrn->input_first? ptrn->input_first : img_view, 1, &ptrl->err_w[0][0], ptrl->prev_n);
			if (ptrn->input_first) {
				for (ptrn_prev = ptrn->in_nets[0]; ptrn_prev < ptrn->in_nets[0] + ptrn->num_in_nets; ptrn_prev++) {
					cblas_sgemv(CblasRowMajor, CblasTrans, ptrl->n, ptrn_prev->arr[ptrn_prev->num_layers-1].n,
							1, &ptrl->w[0][next_col], ptrl->prev_n, 
							ptrl->err_b, 1, 
							0, ptrn_prev->arr[ptrn_prev->num_layers-1].err_b, 1);
					next_col += ptrn_prev->arr[ptrn_prev->num_layers-1].n;
				}
			} else
				img_view += ptrl->prev_n;
		}
	}
}

#define RATE 0.1
#define MOMENTUM 0.3
#define NEW_CHANGE(ERR, CHA) ERR * RATE + MOMENTUM * CHA

void apply_backpr()
{
	struct net *ptrn;
	struct layer *ptrl;
	float change;
	int i, j;
		
	for (ptrn=bignet.arr; ptrn < bignet.arr + bignet.num_nets; ptrn++)
		for (ptrl=ptrn->arr; ptrl < ptrn->arr + ptrn->num_layers; ptrl++)
			for (i=0; i < ptrl->n; i++) {
				for (j=0; j < ptrl->prev_n; j++) {
					change = NEW_CHANGE(ptrl->err_w[i][j], ptrl->change_w[i][j]);
					ptrl->w[i][j] -= change;
					ptrl->change_w[i][j] = change;
				}
				change = NEW_CHANGE(ptrl->err_b[i], ptrl->change_b[i]);
				ptrl->b[i] -= change;
				ptrl->change_b[i] = change;
			}
}

static float img[PIXEL_QTT];
static points[8][2];

#define WHITER 0.5
#define FIND_EDGES \
			if (img[i * DIM_POOL + j] > WHITER) { \
				points[k][0] = i; points[k][1] = j; \
				k++; \
				found = 1; \
				break; \
			} \
		if (found) { \
			found = 0; \
			break; \
		} \
	}

static void metadata(img_view)
float *img_view;
{
	int i, j, k, found;
	
	k = found = 0;
	for (i=0; i < DIM_POOL; i++) {
		for (j=0; j < DIM_POOL; j++)
			FIND_EDGES
	for (i=0; i < DIM_POOL; i++) {
		for (j=DIM_POOL-1; j >= 0; j--)
			FIND_EDGES
	for (i=DIM_POOL-1; i >= 0; i--) {
		for (j=0; j < DIM_POOL; j++)
			FIND_EDGES
	for (i=DIM_POOL-1; i >= 0; i--) {
		for (j=DIM_POOL-1; j >= 0; j--)
			FIND_EDGES
	for (j=0; j < DIM_POOL; j++) {
		for (i=0; i < DIM_POOL; i++)
			FIND_EDGES
	for (j=0; j < DIM_POOL; j++) {
		for (i=DIM_POOL-1; i >= 0; i--)
			FIND_EDGES
	for (j=DIM_POOL-1; j >= 0; j--) {
		for (i=0; i < DIM_POOL; i++)
			FIND_EDGES
	for (j=DIM_POOL-1; j >= 0; j--) {
		for (i=DIM_POOL-1; i >= 0; i--)
			FIND_EDGES
	k = DIM_IMG * DIM_IMG;
	for (i=0; i < 8; i++)
		for (j=i+1; j < 8; j++)
			img_view[k++] = (abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])) / (DIM_IMG * 2.0);
}

static float blur[3][3] = {
	{0.0625, 0.125, 0.0625},
	{0.125 , 0.25 , 0.125},
	{0.0625, 0.125, 0.0625}
};
/*
static edge_right_v[3][3] = {
	{-1, 0, 1},
	{-1, 0, 1},
	{-1, 0, 1}
};
*/
static void convolution(img_view)
float *img_view;
{
	int i, j, k, l;
	float counter;
	
	for (i=1; i < DIM_POOL-1; i++)
		for (j=1; j < DIM_POOL-1; j++) {
			counter = 0;
			for (k=0; k < 3; k++)
				for (l=0; l < 3; l++)
					counter += img[(i-1+k) * DIM_POOL + (j-1+l)] * blur[k][l];
			img_view[(i-1) * DIM_IMG + (j-1)] = counter;
		}
}

#define END png_destroy_read_struct(&png, &info, NULL); fclose(fp);

read_png_file(name, img_view, verbose)
char name[];
float *img_view;
{
	FILE *fp;
	unsigned char header[8];
	png_structp png;
	png_infop info;
	png_bytepp rows;
	png_byte color_type;
	int height;
	int i, j, k, l;
	float sum;
	
	if (!img_view) {
		fputs("[read_png_file] null image array\n", stderr);
		return 1;
	}
	if (!(fp = fopen(name, "rb"))) {
		fprintf(stderr, "[read_png_file] File %s could not be opened for reading\n", name);
		return 2;
	}
	fread(header, 1, 8, fp);
	if (png_sig_cmp(header, 0, 8)) {
		fprintf(stderr, "[read_png_file] File %s is not recognized as a PNG image\n", name);
		return 3;
	}
	if (!(png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL))) {
		END
		fputs("[read_png_file] png_create_read_struct failed\n", stderr);
		return 4;
	}
	if (!(info = png_create_info_struct(png))) {
		END
		fputs("[read_png_file] png_create_info_struct failed\n", stderr);
		return 5;
	}
	if (setjmp(png_jmpbuf(png))) {
		END
		fputs("[read_png_file] init_io failed\n", stderr);
		return 6;
	}
	png_init_io(png, fp);
	png_set_sig_bytes(png, 8);
	png_read_info(png, info);
	height = png_get_image_height(png, info);
	color_type = png_get_color_type(png, info);
	if (verbose)
		printf("WIDTH: %d HEIGHT: %d COLOR_TYPE: %d BIT_DEPTH: %d INTERLACE_HANDLING: %d\n", png_get_image_width(png, info),
                                                                                         height,
                                                                                         color_type,
                                                                                         png_get_bit_depth(png, info),
                                                                                         png_set_interlace_handling(png));
	if (color_type == PNG_COLOR_TYPE_PALETTE) {
		png_set_palette_to_rgb(png);
		png_set_rgb_to_gray(png, 1, 0, 0);
	}
	png_read_update_info(png, info);
	if (setjmp(png_jmpbuf(png))) {
		END
		fputs("[read_png_file] read_image failed\n", stderr);
		return 7;
	}
	rows = (png_bytepp) malloc(sizeof(png_bytep) * height);
	for (i=0; i < height; i++)
		rows[i] = (png_bytep) malloc(png_get_rowbytes(png, info));
	png_read_image(png, rows);
	for (i=0; i < DIM_POOL; i++) 
		for (j=0; j < DIM_POOL; j++) {
			sum = 0;
			for (k=0; k < POOL_LEN; k++)
				for (l=0; l < POOL_LEN; l++)
					sum += rows[i * POOL_LEN + k][j * POOL_LEN + l];
			img[i * DIM_POOL + j] = 1 - sum / (POOL_LEN * POOL_LEN) / 255;
		}
	for (i=0; i < height; i++)
		free(rows[i]);
	free(rows);
	END
	metadata(img_view);
	convolution(img_view);
	return 0;
}

#define FLOAT_RANDOM_WEIGHT ((float)rand() / RAND_MAX - 0.5)
#define FLOAT_RANDOM_BIAS (FLOAT_RANDOM_WEIGHT * 2)

void init_net_topology(nets, n, verbose)
struct create_network nets[];
{
	struct net *ptrn;
	struct create_network *ptrc;
	int i, j, k;
	unsigned *amount;
	
	bignet.num_nets = n;
	bignet.back_on = 0;
	if (verbose)
		puts("Allocating memory to the neural network...");
	bignet.arr = malloc(sizeof(struct net) * n);
	for (ptrn=bignet.arr, ptrc=nets, i=0; i < n; ptrn++, ptrc++, i++) {
		ptrn->arr = malloc(sizeof(struct layer) * ptrc->num_layers);
		ptrn->num_layers = ptrc->num_layers;
		ptrn->out_id = ptrc->output;
		ptrn->input_first = NULL;
		for (j=0; j < ptrn->num_layers; j++) {
			ptrn->arr[j].n = ptrc->neurons_per_layer[j];
			ptrn->arr[j].prev_n = j? ptrc->neurons_per_layer[j-1] : ptrc->num_input;
			ptrn->arr[j].w = malloc(sizeof(float*) * ptrn->arr[j].n);
			*(ptrn->arr[j].w) = malloc(sizeof(float) * ptrn->arr[j].n * ptrn->arr[j].prev_n);
			for (k=1; k < ptrn->arr[j].n; k++)
				ptrn->arr[j].w[k] = *(ptrn->arr[j].w) + k * ptrn->arr[j].prev_n;
			ptrn->arr[j].b = malloc(sizeof(float) * ptrn->arr[j].n);
			ptrn->arr[j].z = malloc(sizeof(float) * ptrn->arr[j].n);
			if (j < ptrn->num_layers-1 || ptrc->output == -1)
				ptrn->arr[j].a = malloc(sizeof(float) * ptrn->arr[j].n);
		}	
		if ((ptrn->output_original = ptrn->out_id == -1)) {
			network_output = ptrn->arr[ptrn->num_layers-1].a;
			bignet.num_classes = ptrn->arr[ptrn->num_layers-1].n;
		}
	}
	amount = calloc(bignet.num_nets - 1, sizeof(unsigned));
	for (ptrn=bignet.arr, ptrc=nets, i=0; ptrn < bignet.arr + bignet.num_nets; ptrn++, ptrc++, i++) {
		ptrn->in_nets = malloc(sizeof(struct net*) * (bignet.num_nets - 1));
		ptrn->num_in_nets = 0;
	}
	for (ptrn=bignet.arr, ptrc=nets, i=0; ptrn < bignet.arr + bignet.num_nets - 1; ptrn++, ptrc++, i++) {
		bignet.arr[ptrn->out_id].in_nets[bignet.arr[ptrn->out_id].num_in_nets++] = ptrn;
		amount[ptrn->out_id] += ptrn->arr[ptrn->num_layers-1].n * sizeof(float);
	}
	for (ptrn=bignet.arr, i=0; ptrn < bignet.arr + bignet.num_nets; ptrn++, i++)
		if (amount[i]) {
			ptrn->in_nets[0]->arr[ptrn->in_nets[0]->num_layers-1].a = malloc(amount[i]);
			ptrn->in_nets[0]->output_original = 1;
			ptrn->input_first = ptrn->in_nets[0]->arr[ptrn->in_nets[0]->num_layers-1].a;
			for (j=1; j < ptrn->num_in_nets; j++)
				ptrn->in_nets[j]->arr[ptrn->in_nets[j]->num_layers-1].a = ptrn->in_nets[j-1]->arr[ptrn->in_nets[j-1]->num_layers-1].a + ptrn->in_nets[j]->arr[ptrn->in_nets[j]->num_layers-1].n;
		}
	free(amount);
	if (verbose)
		puts("Done.");
}

void init_random_weights()
{
	struct net *ptrn;
	struct layer *ptrl;
	int i, j;
	
	for (ptrn=bignet.arr; ptrn < bignet.arr + bignet.num_nets; ptrn++) {
		for (ptrl=ptrn->arr; ptrl < ptrn->arr + ptrn->num_layers; ptrl++)
			for (i=0; i < ptrl->n; i++) {
				for (j=0; j < ptrl->prev_n; j++)
					ptrl->w[i][j] = FLOAT_RANDOM_WEIGHT;
				ptrl->b[i] = FLOAT_RANDOM_BIAS;
			}
	}
}

void load_weights(verbose)
{
	FILE *fp;
	int i, j;
	struct create_network *nets, *ptrc;
	struct net *ptrn;
	struct layer *ptrl;
	
	if (verbose)
		puts("Loading weights and biases to the network from: weights...");
	if (!(fp = fopen("weights", "r"))) {
		fputs("[load_weights] File weights could not be opened for reading.\n", stderr);
		return;
	}
	fscanf(fp, "%x\n", &i);
	nets = malloc(sizeof(struct create_network) * i);
	for (ptrc=nets; ptrc < nets + i; ptrc++) {
		fscanf(fp, "%x %hhx %hd %x", &ptrc->num_layers, &ptrc->source, &ptrc->output, &ptrc->num_input);
		ptrc->neurons_per_layer = malloc(sizeof(short) * ptrc->num_layers);
		for (j=0; j < ptrc->num_layers; j++)
			fscanf(fp, " %x", &ptrc->neurons_per_layer[j]);
		fgetc(fp);
	}
	init_net_topology(nets, i, verbose);
	for (ptrc=nets; ptrc < nets + i; ptrc++)
		free(ptrc->neurons_per_layer);
	free(nets);
	for (ptrn=bignet.arr; ptrn < bignet.arr + bignet.num_nets; ptrn++)
		for (ptrl=ptrn->arr; ptrl < ptrn->arr + ptrn->num_layers; ptrl++)
			for (i=0; i < ptrl->n; i++) {
				for (j=0; j < ptrl->prev_n; j++)
					fscanf(fp, "%a\n", &ptrl->w[i][j]);
				fscanf(fp, "%a\n", &ptrl->b[i]);
			}
	fclose(fp);
	if (verbose)
		puts("Done.");
}

void save_weights()
{
	FILE *fp;
	struct net *ptrn;
	struct layer *ptrl;
	int i, j, k;
	
	puts("Saving network weights and biases to: weights...");
	if (!(fp = fopen("weights", "w"))) {
		fputs("[save_weights] Could not create network file. Discarding changes made.\n", stderr);
		return;
	}
	fprintf(fp, "%x\n", bignet.num_nets);
	for (ptrn=bignet.arr; ptrn < bignet.arr + bignet.num_nets; ptrn++) {
		fprintf(fp, "%x %hhx %hd %x", ptrn->num_layers, ptrn->num_in_nets? 0 : 1, ptrn->out_id, ptrn->arr->prev_n);
		for (ptrl=ptrn->arr; ptrl < ptrn->arr + ptrn->num_layers; ptrl++)
			fprintf(fp, " %x", ptrl->n);
		fputc('\n', fp);
	}
	if (bignet.back_on)
		end_backpr();
	for (ptrn=bignet.arr; ptrn < bignet.arr + bignet.num_nets; ptrn++) {
		for (i=0, ptrl=ptrn->arr; i < ptrn->num_layers; ptrl++, i++) {
			for (j=0; j < ptrl->n; j++) {
				for (k=0; k < ptrl->prev_n; k++)
					fprintf(fp, "%a\n", ptrl->w[j][k]);
				fprintf(fp, "%a\n", ptrl->b[j]);
			}
			free(ptrl->w[0]);
			free(ptrl->w); free(ptrl->b); free(ptrl->z);
			if (ptrn->output_original || i < ptrn->num_layers-1)
				free(ptrl->a);
		}
		free(ptrn->arr);
	}
	fclose(fp);
	puts("Done.");
}

float hit(class, predi, predv)
float class[], *predv;
unsigned char *predi;
{
	int i, bigi;
	float big;
	
	for (big=i=0; i < bignet.num_classes; i++)
		if (big < network_output[i]) {
			big = network_output[i];
			bigi = i;
		}
	if (predi)
		*predi = bigi; 
	if (predv)
		*predv = big; 
	return class? class[bigi] : 0;
}

float cross_entropy(class)
float class[];
{
	int i;
	float c;
	
	for (c=i=0; i < bignet.num_classes; i++)
		c += class[i] * log(network_output[i]);
	return -c;
}
