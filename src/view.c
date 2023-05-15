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
#include <sys/stat.h>
#include <dirent.h>

struct {
	unsigned view : 1;
	unsigned rec_dirs : 1;
	unsigned read_dirs : 1;
	unsigned all_prev : 1;
	unsigned histogram : 1;
	unsigned write_view : 1;
	unsigned verbose : 1;
} flags;

cmp_strings(x, y)
void *x, *y;
{
	return strcmp(*(char**)x, *(char**)y);
}

without_points_dirs(f)
struct dirent *f;
{
	return strcmp(f->d_name, ".") && strcmp(f->d_name, "..");
}

char cwd[PATH_MAX];

unsigned *histogram;

#define MAP_CLASS(id) (id > 9? id - 10 + 'A' : id + '0')

main(argc, argv)
char *argv[];
{
	void rec_read_file();
	char **files, *c;
	int i, end;

	if (argc < 2) {
		puts("Usage: view -a -d -v -R image");
		return 1;
	}
	files = malloc(sizeof(char*) * (argc-1));
	end = 0;
	flags.view = 1;
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
				case 'v':
					flags.verbose = 1;
					break;
				case 'H':
					flags.view = 0;
				case 'h':
					flags.histogram = 1;
					break;
				case 'o':
					flags.write_view = 1;
					break;
				default:
					printf("view: illegal option %c\n", *c);
					puts("Usage: view -a -d -v -R image");
					return 2;
				}
		else
			files[end++] = *argv;
	qsort(files, end, sizeof(char*), cmp_strings);
	if (!getcwd(cwd, sizeof(cwd))) {
		perror("[getcwd]");
		return 8;
	}
	if (flags.histogram) {
		histogram = malloc(sizeof(unsigned) * MAX_CLASSES);
		for (i=0; i < MAX_CLASSES; i++)
			histogram[i] = 0;
	}
	load_weights(0);
	for (i=0; i < end; i++)
		rec_read_file(files[i], cwd, 0);
	if (flags.histogram)
		for (i=0; i < MAX_CLASSES; i++)
			printf("%c %hd\n", MAP_CLASS(i), histogram[i]);
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

	if (!flags.read_dirs) {
		read_img(name, level);
		return;
	}
	if (!stat(name, &filestat)) {
		if (S_ISDIR(filestat.st_mode)) {
			if ((n = scandir(name, &files_in_dir, without_points_dirs, alphasort)) < 0)
				perror("scandir");
			else {
				if (flags.view) {
					ALIGN
					printf("%s:\n", name);
				}
				chdir(name);
				for (i=0; i < n; i++) {
					if (flags.rec_dirs)
						rec_read_file(files_in_dir[i]->d_name, "..", level+1);
					else if (!stat(files_in_dir[i]->d_name, &filestat)) {
						if (S_ISREG(filestat.st_mode))
							read_img(files_in_dir[i]->d_name, level+1);
					} else
						fprintf(stderr, "Failed to stat %s\n", files_in_dir[i]->d_name);
					free(files_in_dir[i]);
				}
				free(files_in_dir);
				chdir(prev_wd);
			}
		} else 
			read_img(name, level);
	} else
		fprintf(stderr, "Failed to stat %s\n", name);
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
	float pred;
	int error, i, j;
	unsigned char class;

	if ((error = read_png_file(name, &img, flags.verbose))) {
		fprintf(stderr, "Error: %d. Cannot read file: %s\n", error, name);
		return;
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
