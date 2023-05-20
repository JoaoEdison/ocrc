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
#include <unistd.h>
#include <dirent.h>
#include <sys/time.h>
#include <pthread.h>
#include <string.h>

struct node {
	float value[INPUT_QTT];
	struct node *next;
};

char cwd[PATH_MAX];

main(argc, argv)
char *argv[];
{
	float (*metric)();
	char *ptrc;
	int error;
	
	if (argc < 2) {
		puts("Usage: training epochs -metric=X");
		return 7;
	}
	load_weights(1);
	if (!getcwd(cwd, sizeof(cwd))) {
		perror("[getcwd]");
		return 8;
	}
	metric = NULL;
	if (argc > 2) {
		if (strncmp(argv[2], "-metric=", 8))
			printf("training: illegal option %s\n", argv[2]);
		else {
			ptrc = argv[2]+8;
			if (!(strcmp("accuracy", ptrc) && strcmp("a", ptrc)))
				metric = hit;
			else if (!(strcmp("cross-entropy", ptrc) && strcmp("cross", ptrc) && strcmp("c", ptrc)))
				metric = cross_entropy;
			else if (strcmp("none", ptrc) && strcmp("n", ptrc))
				puts("Non-existent or unsupported metric");
		}
	}
	if ((error = train(atoi(argv[1]), metric)))
		return error;
	save_weights();
}

read_paths(fname, n, views)
char fname[]; 
int *n;
struct node views[];
{
	FILE *fp;
	DIR *dirp;
	struct dirent *file;
	int i, j, c;
	char buf[MAX_CLASSES][NAME_MAX];
	struct node *ptrnode;
	char first;
	
	if (!(fp = fopen(fname, "r"))) {
		fprintf(stderr, "[read_paths] Unable to open datasets path file %s\n", fname);
		return 9;
	}
	for (c = fgetc(fp), j=i=0; c != EOF && i < MAX_CLASSES; c = fgetc(fp)) {
		if (c == '\n') {
			buf[i][j] = '\0';
			i++;
			j=0;
		} else if (j < NAME_MAX) {
			buf[i][j++] = c;
		} else {
			buf[i][j-1] = '\0';
			break;
		}
	}
	fclose(fp);
	for (*n=i=0; i < MAX_CLASSES; i++) {
		if (!(dirp = opendir(buf[i]))) {
			fprintf(stderr, "[read_paths] Unable to open directory %s\n", buf[i]);
			return 10;
		}
		chdir(buf[i]);
		ptrnode = &views[i];
		first = 1;
		while ((file = readdir(dirp)))
			if (file->d_type == DT_REG) {
				(*n)++;
				if (!first) {
					ptrnode->next = malloc(sizeof(struct node));
					ptrnode = ptrnode->next;
				}
				first = 0;
				if (read_png_file(file->d_name, &ptrnode->value, 0)) {
					fputs("[read_png_file] Error\n", stderr);
					break;
				}
			}
		ptrnode->next = NULL;
		closedir(dirp);
	}
	chdir(cwd);
	return 0;
}

pthread_mutex_t mutex;
char state;

train(epochs, metric_fn)
float (*metric_fn)();
{
	void *stop_training(), print_fomat_seconds();
	float avg();
	struct node train_views[MAX_CLASSES], test_views[MAX_CLASSES], *ptrnode, *auxnode;
	float class[MAX_CLASSES];
	int i, j, train_count, test_count, error;
	struct timeval begin, end;
	pthread_t tid;
	
	puts("Reading images of training and testing sets...");
	if ((error = read_paths("dataset_paths", &train_count, train_views)))
		return error;
	if ((error = read_paths("test_paths", &test_count, test_views)))
		return error;
	for (i=0; i < MAX_CLASSES-1; i++)
		class[i] = 0;
	if (metric_fn)
		printf("%.2f (test) %.2f (training) [0/%hd]\n", avg(metric_fn, test_count, test_views), avg(metric_fn, train_count, train_views), epochs);
	ini_backpr(train_count);
	state = 's';
	pthread_mutex_init(&mutex, NULL);
	pthread_create(&tid, NULL, stop_training, NULL);
	gettimeofday(&begin, 0);
	for (i=0; i < epochs; i++) {
		clear_backpr();
		class[MAX_CLASSES-1] = 0;
		for (j=0; j < MAX_CLASSES; j++) {
			if (j)
				class[j-1] = 0;
			class[j] = 1;
			ptrnode = &train_views[j];
			while (ptrnode) {
				run(ptrnode->value);
				backpr(class, ptrnode->value);
				ptrnode = ptrnode->next;
			}
		}
		apply_backpr();
		if (metric_fn)
			printf("%.2f (test) %.2f (training) [%hd/%hd]\n", avg(metric_fn, test_count, test_views), avg(metric_fn, train_count, train_views), i+1, epochs);
		if (state != 's')
			break;
		if (!i) {
			gettimeofday(&end, 0);
			printf("Approximate training time: ");
			print_fomat_seconds((end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) / 1e6) * (epochs - 1));
		}
	}
	pthread_mutex_lock(&mutex);
	if (state == 's')
		pthread_cancel(tid);
	pthread_join(tid, NULL);
	pthread_mutex_unlock(&mutex);
	pthread_mutex_destroy(&mutex);
	end_backpr();
	for (i=0; i < MAX_CLASSES; i++) {
		ptrnode = test_views[i].next;
		while (ptrnode) {
			auxnode = ptrnode;
			ptrnode = ptrnode->next;
			free(auxnode);
		}
		ptrnode = train_views[i].next;
		while (ptrnode) {
			auxnode = ptrnode;
			ptrnode = ptrnode->next;
			free(auxnode);
		}
	}
	return 0;
}

float avg(metric_fn, n, views)
float (*metric_fn)();
struct node views[];
{
	float class[MAX_CLASSES];
	struct node *ptrnode;
	float sum;
	int i;
	
	for (i=0; i < MAX_CLASSES; i++)
		class[i] = 0;
	for (sum=i=0; i < MAX_CLASSES; i++) {
		if (i)
			class[i-1] = 0;
		class[i] = 1;
		ptrnode = &views[i];
		while (ptrnode) {
			run(ptrnode->value);
			sum += metric_fn(class, NULL, NULL);
			ptrnode = ptrnode->next;
		}
	}
	return sum / (float) n;
}

void *stop_training()
{
	int c;
	
	puts("Press 'q' to end training");
	for (c = getchar(); c != 'q' && c != 'Q'; c = getchar());
	pthread_mutex_lock(&mutex);
	state = 'E';
	pthread_mutex_unlock(&mutex);
}

void print_fomat_seconds(seconds)
float seconds;
{
	int units, i;
	
	for (i=2; i; i--) {
		printf("%02d:", (units = seconds / pow(60, i)));
		if (units)
			seconds -= pow(60, i) * units;
	}
	printf("%02d\n", (int)seconds);
}
