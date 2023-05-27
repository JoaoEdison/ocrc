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
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>

cmp_strings(x, y)
void *x, *y;
{
	return strcmp(*(char**)x, *(char**)y);
}

struct {
	unsigned view : 1;
	unsigned rec_dirs : 1;
	unsigned read_dirs : 1;
	unsigned all_prev : 1;
	unsigned histogram : 1;
	unsigned write_view : 1;
	unsigned verbose : 1;
} flags;

char cwd[PATH_MAX];
unsigned *histogram, count;

#define MAP_CLASS(id) (id > 9? id - 10 + 'A' : id + '0')

main(argc, argv)
char *argv[];
{
	void rec_read_file();
	char **files, *c;
	struct stat st;
	int i, end;

	if (argc < 2) {
		puts("Usage: view -a -d -h -o -v -R -H image");
		return 1;
	}
	end = 0;
	flags.view = 1;
	files = malloc(sizeof(char*) * (argc-1));
	while (--argc)
		if ((*++argv)[0] == '-')
			for (c=argv[0]+1; *c; c++)
				switch (*c) {
				case 'a':
					flags.all_prev = 1;
					break;
				case 'R':
					flags.rec_dirs = 1;
				case 'd':
					flags.read_dirs = 1;
					break;
				case 'H':
					flags.view = 0;
				case 'h':
					flags.histogram = 1;
					break;
				case 'o':
					flags.write_view = 1;
					break;
				case 'v':
					flags.verbose = 1;
					break;
				default:
					printf("view: illegal option %c\n", *c);
					puts("Usage: view -a -d -h -o -v -R -H image");
					return 2;
				}
		else
			files[end++] = *argv;
	qsort(files, end, sizeof(char*), cmp_strings);
	if (!getcwd(cwd, sizeof(cwd))) {
		perror("[main] getcwd");
		return 3;
	}
	if (flags.histogram)
		histogram = calloc(MAX_CLASSES, sizeof(unsigned));
	if (flags.write_view && stat("views_output", &st))
		if (mkdir("views_output", 0777)) {
			perror("[main] mkdir");
			fprintf(stderr, "Failed to create directory: views_output\n");
			return 4;
		}
	load_weights(0);
	count = 0;
	for (i=0; i < end; i++)
		rec_read_file(files[i], cwd, 0);
	free(files);
	if (flags.histogram)
		for (i=0; i < MAX_CLASSES; i++)
			printf("%c %u\n", MAP_CLASS(i), histogram[i]);
	return 0;
}

without_points_dirs(f)
struct dirent *f;
{
	return strcmp(f->d_name, ".") && strcmp(f->d_name, "..");
}

#define ALIGN  \
	for (i=0; i < level; i++) \
		putchar('\t');

void rec_read_file(name, prev_wd, level)
char name[], prev_wd[];
{
	void read_img();
	struct dirent **files_in_dir;
	struct stat filestat;
	int i, n;
	
	if (stat(name, &filestat)) {
		perror("[rec_read_file] stat");
		fprintf(stderr, "Failed to stat %s\n", name);
		return;
	}
	if (!flags.read_dirs || S_ISREG(filestat.st_mode)) {
		read_img(name, level);
		return;
	}
	if (!flags.rec_dirs && level)
		return;
	if ((n = scandir(name, &files_in_dir, without_points_dirs, alphasort)) < 0) {
		perror("[rec_read_file] scandir");
		return;
	}
	if (flags.view) {
		ALIGN
		printf("%s:\n", name);
	}
	if (chdir(name)) {
		perror("[rec_read_file] chdir");
		return;
	}
	for (i=0; i < n; i++) {
		rec_read_file(files_in_dir[i]->d_name, "..", level+1);
		free(files_in_dir[i]);
	}
	free(files_in_dir);
	if (chdir(prev_wd))
		perror("[rec_read_file] chdir");
}

struct map_prob {
	char class;
	float vpred;
} probs[MAX_CLASSES];

cmp_floats(x, y)
void *x, *y;
{
	return ((struct map_prob*)x)->vpred < ((struct map_prob*)y)->vpred;
}

void read_img(name, level)
char name[];
{
	void write_png();
	static float img[INPUT_QTT];
	char view_name[PATH_MAX + NAME_MAX];
	float pred;
	int error, i, j, cn;
	unsigned char class;

	if ((error = read_png_file(name, img, flags.verbose))) {
		fprintf(stderr, "[read_img] Error: %d. Cannot read file: %s\n", error, name);
		return;
	}
	if (flags.write_view) {
		count++;
		cn = sprintf(view_name, "%s/views_output/%03dv_", cwd, count);
		for (j=i=0; name[i]; i++)
			if (name[i] == '/')
				j=i+1;
		cn += snprintf(view_name+cn, PATH_MAX + NAME_MAX - cn, "%s", name+j);
		if (cn > PATH_MAX + NAME_MAX)
			sprintf(view_name + PATH_MAX + NAME_MAX - 5, ".png");
		write_png(img, view_name);
	}
	run(img);
	hit(NULL, &class, &pred);
	if (flags.histogram)
		histogram[class]++;
	if (flags.view) {
		ALIGN
		printf("%s: %c -> %.2f\n", name, MAP_CLASS(class), pred);
		if (flags.all_prev) {
			for (j=0; j < MAX_CLASSES; j++) {
				probs[j].class = MAP_CLASS(j);
				probs[j].vpred = network_output[j];
			}
			qsort(probs, MAX_CLASSES, sizeof(struct map_prob), cmp_floats);
			for (j=0; j < MAX_CLASSES; j++) {
				ALIGN
				printf("\t%f -> %c\n", probs[j].vpred, probs[j].class);
			}
		}
	}
}

void write_png(img, name)
float img[];
char name[];
{
        FILE *fp;
        png_structp png;
        png_infop info;
        png_bytepp rows;
        int i, j;
	
	if (!(fp = fopen(name, "wb"))) {
		fprintf(stderr, "[write_png] File %s could not be opened for writing\n", name);
		return;
	}
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
