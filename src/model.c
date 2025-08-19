/*
    OCRC, a AI for optical character recognition written in C
    Copyright (C) 2023-2025 João E. R. Manica
    
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

/* Model configuration functions */

#include "model.h"

#include <stdlib.h>
#include <png.h>
/*1.6.37*/
#include <math.h>

#define WHITER 0.5
#define FIND_EDGES \
			if (img_in[i * dim_in + j] > WHITER) { \
				points[k][0] = i; points[k][1] = j; \
				k++; \
				found = 1; \
				break; \
			} \
                }\
		if (found) { \
			found = 0; \
			break; \
		} \
	}

/* Gets this points, calculates the Manhattan distances beetween
   and divides by the main diagonal. Stores it in the end of img_view.
 *          57
 *          ||
 *          vv
 *      1-> /\ <-2
 *         /  \
 *        /    \
 *       /------\
 *      /        \
 * 3-> /          \ <-4
 *     ^          ^
 *     |          |
 *     6          8
 * Sum the cols, make the mean. Sum the rows, make the mean.
 * Stores both after the images area (FEATURE_QTT * AREA_IMG).
 * */
static void metadata(img_view, img_in, dim_in)
float *img_view, *img_in;
{
	static points[8][2];
	int i, j, k, found;
	
	k = found = 0;
	/*1*/
	for (i=0; i < dim_in; i++) {
		for (j=0; j < dim_in; j++) {
			FIND_EDGES
	/*2*/
	for (i=0; i < dim_in; i++) {
		for (j=dim_in-1; j >= 0; j--) {
			FIND_EDGES
	/*3*/
	for (i=dim_in-1; i >= 0; i--) {
		for (j=0; j < dim_in; j++) {
			FIND_EDGES
	/*4*/
	for (i=dim_in-1; i >= 0; i--) {
		for (j=dim_in-1; j >= 0; j--) {
			FIND_EDGES
	/*5*/
	for (j=0; j < dim_in; j++) {
		for (i=0; i < dim_in; i++) {
			FIND_EDGES
	/*6*/
	for (j=0; j < dim_in; j++) {
		for (i=dim_in-1; i >= 0; i--) {
			FIND_EDGES
	/*7*/
	for (j=dim_in-1; j >= 0; j--) {
		for (i=0; i < dim_in; i++) {
			FIND_EDGES
	/*8*/
	for (j=dim_in-1; j >= 0; j--) {
		for (i=dim_in-1; i >= 0; i--) {
			FIND_EDGES
	k = FEATURE_QTT * AREA_IMG;
	for (i=0; i < dim_in; i++) {
		img_view[k] = 0;
		for (j=0; j < dim_in; j++)
			img_view[k] += img_in[i * dim_in + j];
		img_view[k] /= dim_in;
		k++;
	}
	for (j=0; j < dim_in; j++) {
		img_view[k] = 0;
		for (i=0; i < dim_in; i++)
			img_view[k] += img_in[i * dim_in + j];
		img_view[k] /= dim_in;
		k++;
	}
	for (i=0; i < 8; i++)
		for (j=i+1; j < 8; j++)
			img_view[k++] = (abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])) / (dim_in * 2.0);
}

float *img, *copy_flat;

void config_vision()
{
	copy_flat = malloc(sizeof(float) * WIDTH * HEIGHT);
	if (POOL_LEN)
		img = malloc(sizeof(float) * PIXEL_QTT);
}

#define END png_destroy_read_struct(&png, &info, NULL); fclose(fp);

/* read_png_file:
 *     Reads 'name' using fopen and verifies if it is a PNG file.
 *     The function applies convolution to the image and extracts metadata.
 *     Returns 0 if the file is successfully read and the output is stored in 'img_view'.
 *     Else returns an error value and prints an error message.
 *     If 'verbose' is not zero, it displays the characteristics of the file. 
 *     'img_view' should have a length of INPUT_QTT.
 * */
read_png_file(name, img_view, verbose)
char name[];
float *img_view;
{
    extern float vision_blur_k3[9];
	static unsigned char header[8];
	FILE *fp;
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
	if (fread(header, 1, 8, fp) != 8) {
		fputs("[read_png_file] Header reading error\n", stderr);
		return 3;
	}
	if (png_sig_cmp(header, 0, 8)) {
		fprintf(stderr, "[read_png_file] File %s is not recognized as a PNG image\n", name);
		return 4;
	}
	if (!(png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL))) {
		END
		fputs("[read_png_file] png_create_read_struct failed\n", stderr);
		return 5;
	}
	if (!(info = png_create_info_struct(png))) {
		END
		fputs("[read_png_file] png_create_info_struct failed\n", stderr);
		return 6;
	}
	if (setjmp(png_jmpbuf(png))) {
		END
		fputs("[read_png_file] init_io failed\n", stderr);
		return 7;
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
		return 8;
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
	metadata(img_view, img, DIM_POOL);
	vision_convolution(img_view, img, vision_blur_k3, DIM_POOL, DIM_POOL, DIM_IMG1, 3, 0);
	return 0;
}

void writepng(source)
float source[];
{
	FILE *out;
	png_structp png;
    png_infop info;
	png_bytepp image;
	int i, j;
	static number = 0;
	static char name[11] = "teste .png";
	
	name[5] = ++number + '0';
	if (!(out = fopen(name, "wb"))) {
		fputs("Can't open teste.png\n", stderr);
		exit(1);
	}
	if (!(image = malloc(sizeof(png_bytep) * HEIGHT))){
		fputs("No memory for jpeg convert\n", stderr);
		exit(1);
	}
    for (i=0; i < HEIGHT; i++) {
        image[i] = calloc(WIDTH*4, 1);
		if (!image[i]){
			fputs("No memory for jpeg convert\n", stderr);
			exit(1);
		}
	}
	for (i=0; i < HEIGHT; i++)
		for (j=0; j < WIDTH; j++)
			image[i][j] = source[i * WIDTH + j] * 255;
    png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info = png_create_info_struct(png);
    png_init_io(png, out);
    png_set_IHDR(png, info, WIDTH, HEIGHT, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_ADAM7, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    png_write_image(png, image);
    png_write_end(png, NULL);
	fclose(out);
}
