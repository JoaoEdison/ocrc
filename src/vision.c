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

/* Image processing functions definitions. */

#include "vision.h"

#include <float.h>

float vision_blur_k3[] = {
	0.0625, 0.125, 0.0625,
	0.125 , 0.25 , 0.125,
	0.0625, 0.125, 0.0625
};

float vision_blur_k5[] = {
	0.00390625, 0.015625  , 0.0234375 , 0.015625  ,0.00390625,
	0.015625  , 0.0625    , 0.09375   , 0.0625    ,0.015625  ,
	0.0234375 , 0.09375   , 0.140625  , 0.09375   ,0.0234375 ,
	0.015625  , 0.0625    , 0.09375   , 0.0625    ,0.015625  ,
	0.00390625, 0.015625  , 0.0234375 , 0.015625  ,0.00390625
};

void vision_flattener(in,w,h,lda,out)
float *in, *out;
{
	int i,j;
	
	for (i=0; i < h; i++)
		for (j=0; j < w; j++)
			out[i*w+j] = in[i*lda+j];
}

void vision_convolution(img_view, in_view, kernel, dim_inw, dim_inh, dim_outw, ksize, padding)
float *img_view, *in_view, kernel[];
vision_positive padding;
{
	int i, j, k, l;
	float counter;
	
	if (padding)
		for (i=0; i < dim_inh; i++)
			for (j=0; j < dim_inw; j++) {
				counter = 0;
				for (k=0; k < ksize; k++)
					for (l=0; l < ksize; l++)
						counter += (i-ksize/2 < 0 || j-ksize/2 < 0 || j-ksize/2+l >= dim_inw || i-ksize/2+k >= dim_inh)?
							   BACKGROUNDCOLOR : in_view[(i-ksize/2+k) * dim_inw + (j-ksize/2+l)] * kernel[k*ksize + l];
				img_view[i * dim_outw + j] = counter;
			}
	else
		for (i=ksize/2; i < dim_inh-ksize/2; i++)
			for (j=ksize/2; j < dim_inw-ksize/2; j++) {
				counter = 0;
				for (k=0; k < ksize; k++)
					for (l=0; l < ksize; l++)
						counter += in_view[(i-ksize/2+k) * dim_inw + (j-ksize/2+l)] * kernel[k*ksize + l];
				img_view[(i-ksize/2) * dim_outw + (j-ksize/2)] = counter;
			}
}

void vision_pooling(img_view, in_view, dim_inw, dim_inh, dim_outw, pool_size, avg)
float *img_view, *in_view;
vision_positive avg;
{
	int i, j, k, l;
	float counter;

	for (i=0; i < dim_inh/pool_size; i++)
		for (j=0; j < dim_inw/pool_size; j++) {
			counter = avg? 0 : FLT_MIN;
			for (k=i*pool_size; k < pool_size; k++)
				for (l=j*pool_size; l < pool_size; l++) {
					if (avg)
						counter += in_view[k*dim_inw + l];
					else if (counter < in_view[k*dim_inw + l])
						counter = in_view[k*dim_inw + l];
				}
			img_view[i*dim_outw+j] = counter;
		}
}
