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

/* Image processing functions declarations. */

#ifndef VISION_H
#define VISION_H

#define BACKGROUNDCOLOR 0

typedef unsigned char vision_positive;

extern float vision_blur_k3[9];
extern float vision_blur_k5[25];

void vision_flattener(float *in, int w, int h, int lda,float *out);

void vision_convolution(float *img_view, float *in_view, float kernel[],
                        int dim_inw, int dim_inh, int dim_outw, int ksize,
                        vision_positive padding);

void vision_pooling(float *img_view, float *in_view, int dim_inw, int dim_inh, int dim_outw, int pool_size, vision_positive avg);

#endif
