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
#include <string.h>
#include <ctype.h>
#include <pthread.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/time.h>

char cwd[PATH_MAX];

main(argc, argv)
char *argv[];
{
	void (*method_fn)(), train_all(), train_stochastic(), train_batch();
	float (*metric)();
	char *ptrc;
	int error, epochs, method_n;
	
	if (argc == 1) {
		puts("Usage: training epochs -metric=accuracy/cross-entropy/none -method=stochastic/batch/all [size]");
		return 1;
	}
	if (!getcwd(cwd, sizeof(cwd))) {
		perror("[main] getcwd");
		return 2;
	}
	method_fn = train_all;
	metric = NULL;
	method_n = epochs = 0;
	while (--argc) {
		if ((*++argv)[0] == '-') {
			ptrc = *argv + 8;
			if (!strncmp(*argv, "-metric=", 8)) {
				if (!(strcmp("accuracy", ptrc) && strcmp("a", ptrc)))
					metric = hit;
				else if (!(strcmp("cross-entropy", ptrc) && strcmp("cross", ptrc) && strcmp("c", ptrc)))
					metric = cross_entropy;
				else if (strcmp("none", ptrc) && strcmp("n", ptrc))
					puts("Non-existent or unsupported metric");
			} else if (!strncmp(*argv, "-method=", 8)) {
				if (!(strcmp("stochastic", ptrc) && strcmp("s", ptrc))) {
					method_n = 1;
					method_fn = train_stochastic;
				} else if (!(strcmp("batch", ptrc) && strcmp("b", ptrc))) {
					method_n = atoi(*++argv);
					argc--;
					method_fn = train_batch;
				} else if (strcmp("all", ptrc) && strcmp("a", ptrc))
					puts("Non-existent or unsupported method");
			} else {
				printf("training: illegal option %s\n", *argv);
				return 3;
			}
		} else if (isdigit(*argv[0]))
			epochs = atoi(*argv);
		else {
			printf("training: illegal option %s\n", *argv);
			return 3;
		}
	}
	load_weights(1);
	if ((error = train(epochs, method_n, method_fn, metric, "dataset_paths", "test_paths")))
		return error;
	save_weights();
	return 0;
}

struct vector_view {
	float **arr;
	int end, size;
};

void train_all(train, class)
struct vector_view train[];
float class[];
{
	int i, j;

	for (i=0; i < MAX_CLASSES; i++, class[i-1] = 0) {
		class[i] = 1;
		for (j=0; j < train[i].end; j++) {
			run(train[i].arr[j]);
			backpr(class, train[i].arr[j]);
		}
	}
}

void train_stochastic(train, class)
struct vector_view train[];
float class[];
{
	int i, j;
	struct vector_view *ptrv;

	i = rand() % MAX_CLASSES;
	class[i] = 1;
	ptrv = &train[i];
	j = rand() % ptrv->end;
	run(ptrv->arr[j]);
	backpr(class, ptrv->arr[j]);
	class[i] = 0;
}

batch_size;

void train_batch(train, class)
struct vector_view train[];
float class[];
{
	void train_stochastic();
	int i;

	for (i=0; i < batch_size; i++)
		train_stochastic(train, class);
}

read_paths(fname, n, views)
char fname[]; 
int *n;
struct vector_view views[MAX_CLASSES];
{
	char buf[MAX_CLASSES][PATH_MAX];
	FILE *fp;
	DIR *dirp;
	struct dirent *file;
	struct vector_view *ptrv;
	int i, j, c, error;
	
	if (!(fp = fopen(fname, "r"))) {
		fprintf(stderr, "[read_paths] Unable to open datasets path file %s\n", fname);
		return 4;
	}
	for (c = fgetc(fp), j=i=0; c != EOF && i < MAX_CLASSES; c = fgetc(fp)) {
		if (c == '\n') {
			buf[i][j] = '\0';
			i++;
			j=0;
		} else if (j < PATH_MAX) {
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
			return 5;
		}
		if (chdir(buf[i])) {
			perror("[read_paths] chdir");
			return 6;
		}
		ptrv = views + i;
		ptrv->arr = malloc(sizeof(float*) * 3);
		ptrv->end = 0;
		ptrv->size = 3;
		while ((file = readdir(dirp)))
			if (file->d_type == DT_REG) {
				ptrv->arr[ptrv->end] = malloc(sizeof(float) * INPUT_QTT);
				if (ptrv->end == ptrv->size-1)
					ptrv->arr = realloc(ptrv->arr, sizeof(float*) * (ptrv->size += 16));
				if ((error = read_png_file(file->d_name, ptrv->arr[ptrv->end], 0))) {
					fprintf(stderr, "[read_paths] Error: %d. Cannot read file: %s\n", error, file->d_name);
					break;
				}
				ptrv->end++;
				(*n)++;
			}
		closedir(dirp);
	}
	if (chdir(cwd)) {
		perror("[read_paths] chdir");
		return 7;
	}
	return 0;
}

pthread_mutex_t mutex;
char state;

#define CLOCK 20

train(epochs, method_n, method_fn, metric_fn, fname_training, fname_test)
void (*method_fn)();
float (*metric_fn)();
char fname_training[], fname_test[];
{
	void *stop_training(), print_fomat_seconds();
	float avg();
	struct vector_view train_views[MAX_CLASSES], test_views[MAX_CLASSES];
	float class[MAX_CLASSES];
	int i, j, train_count, test_count, error;
	struct timeval begin, end;
	pthread_t tid;
	
	printf("Reading images of training set %s...\n", fname_training);
	if ((error = read_paths(fname_training, &train_count, train_views)))
		return error;
	puts("Done.");
	printf("Reading images of test set %s...\n", fname_test);
	if ((error = read_paths(fname_test, &test_count, test_views)))
		return error;
	puts("Done.");
	printf("%u training images\n", train_count);
	printf("%u test images\n", test_count);
	if (metric_fn)
		printf("%.3f (test) %.3f (training) [0/%d]\n", avg(metric_fn, test_count, test_views), avg(metric_fn, train_count, train_views), epochs);
	for (i=0; i < MAX_CLASSES-1; i++)
		class[i] = 0;
	state = 's';
	if (!method_n)
		method_n = train_count;
	else
		srand(time(NULL));
	batch_size = method_n;
	ini_backpr(method_n);
	pthread_mutex_init(&mutex, NULL);
	pthread_create(&tid, NULL, stop_training, NULL);
	gettimeofday(&begin, 0);
	for (i=0; i < epochs; i++) {
		clear_backpr();
		method_fn(train_views, class);
		apply_backpr();
		if ((metric_fn && method_n != 1) || (metric_fn && i % CLOCK == 0))
			printf("%.3f (test) %.3f (training) [%d/%d]\n", avg(metric_fn, test_count, test_views), avg(metric_fn, train_count, train_views), i+1, epochs);
		if (state != 's')
			break;
		if ((method_n != 1 && !i) || (method_n == 1 && i == CLOCK-1)) {
			gettimeofday(&end, 0);
			printf("Approximate training time: ");
			print_fomat_seconds((end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) / 1e6) * (
							method_n != 1? epochs - 1 :
							(float)(epochs - CLOCK) / (float)CLOCK
						));
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
		for (j=0; j < test_views[i].end; j++)
			free(test_views[i].arr[j]);
		for (j=0; j < train_views[i].end; j++)
			free(train_views[i].arr[j]);
		free(test_views[i].arr);
		free(train_views[i].arr);
	}
	return 0;
}

float avg(metric_fn, n, views)
float (*metric_fn)();
struct vector_view views[];
{
	float class[MAX_CLASSES];
	float sum;
	int i, j;
	
	for (i=0; i < MAX_CLASSES; i++)
		class[i] = 0;
	for (sum=i=0; i < MAX_CLASSES; i++) {
		if (i)
			class[i-1] = 0;
		class[i] = 1;
		for (j=0; j < views[i].end; j++) {
			run(views[i].arr[j]);
			sum += metric_fn(class, NULL, NULL);
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
