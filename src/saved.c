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

void write_png(img)
float img[];
{
        FILE *fp;
        png_structp png;
        png_infop info;
        png_bytepp rows;
        int i, j;

        fp = fopen("test.png", "wb");
        png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        info = png_create_info_struct(png);
        png_init_io(png, fp);
        png_set_IHDR(png, info, DIM_IMG, DIM_IMG, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_ADAM7, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png, info);
        rows = (png_bytepp) malloc(sizeof(png_bytep) * DIM_IMG);
        for (i=0; i < DIM_IMG; i++)
                rows[i] = (png_bytep) malloc(DIM_IMG * 4);
        for (i=0; i < DIM_IMG; i++)
                for (j=0; j < DIM_IMG; j++)
                        rows[i][j] = (png_byte) (img[i * DIM_IMG + j] * 255);
        png_write_image(png, rows);
        png_write_end(png, NULL);
        for (i=0; i < DIM_IMG; i++)
                free(rows[i]);
	free(rows);
        fclose(fp);
}
