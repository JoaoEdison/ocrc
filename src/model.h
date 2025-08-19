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

/* Model configuration constants */

#ifndef MODEL_H
#define MODEL_H

#define WIDTH 128
#define HEIGHT 128
#define POOL_LEN 8
#define DIM_POOL (WIDTH / POOL_LEN)
#define PIXEL_QTT (DIM_POOL * DIM_POOL)
#define DIM_IMG1 (DIM_POOL - 2)
#define DIM_IMGL DIM_IMG1
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

/*Last layer activation function*/
#define LAST_ACTIVATION_FN \
    do {\
        float sum;\
        \
        for (sum=i=0; i < ptrl->n; i++) {\
            ptrl->a[i] = exp(ptrl->z[i]);\
            sum += ptrl->a[i]; \
        }\
        for (i=0; i < ptrl->n; i++)\
            ptrl->a[i] /= sum;\
        /*
        for (i=0; i < ptrl->n; i++)\
            ptrl->a[i] = ACTIVATION_FN(ptrl->z[i]);\
        */\
    } while (0);

#endif
