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
	unsigned read_dirs : 1;
	unsigned all_prev : 1;
	unsigned histogram : 1;
	unsigned write_view : 1;
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

main(argc, argv)
char *argv[];
{
	void read_img();
	char **files, *c;
	int i, j, n, end;
	struct stat filestat;
	struct dirent **files_in_dir;

	if (argc < 2) {
		puts("Usage: view -d -a -h image");
		return 1;
	}
	files = malloc(sizeof(char*) * (argc-1));
	end = 0;
	while (--argc)
		if ((*++argv)[0] == '-')
			for (c=argv[0]+1; *c; c++)
				switch (*c) {
				case 'a':
					flags.all_prev = 1;
					break;
				case 'd':
					flags.read_dirs = 1;
					break;
				case 'h':
					flags.histogram = 1;
					break;
				case 'o':
					flags.write_view = 1;
					break;
				default:
					printf("view: illegal option %c\n", *c);
					puts("Usage: view -d -a -h image");
					return 2;
				}
		else
			files[end++] = *argv;
	qsort(files, end, sizeof(char*), cmp_strings);
	if (!getcwd(cwd, sizeof(cwd))) {
		perror("[getcwd]");
		return 8;
	}
	load_weights(0);
	for (i=0; i < end; i++) {
		if (!flags.read_dirs) {
			read_img(files[i], 0);
			continue;
		}
		if (!stat(files[i], &filestat)) {
			if (S_ISDIR(filestat.st_mode)) {
				if ((n = scandir(files[i], &files_in_dir, without_points_dirs, alphasort)) < 0)
					perror("scandir");
				else {
					printf("%s:\n", files[i]);
					chdir(files[i]);
					for (j=0; j < n; j++) {
						read_img(files_in_dir[j]->d_name, 1);
						free(files_in_dir[j]);
					}
					free(files_in_dir);
					chdir(cwd);
				}
			} else 
				read_img(files[i], 0);
		} else
			fprintf(stderr, "Failed to stat %s\n", files[i]);
	}
}

#define ALIGN  \
	for (i=0; i < level; i++) \
		putchar('\t');

#define MAP_CLASS(id) (id > 9? id - 10 + 'A' : id + '0')

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

	if ((error = read_png_file(name, &img, 0))) {
		fprintf(stderr, "Error: %d. Cannot read file: %s\n", error, name);
		return;
	}
	run(img);
	hit(NULL, &class, &pred);
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
