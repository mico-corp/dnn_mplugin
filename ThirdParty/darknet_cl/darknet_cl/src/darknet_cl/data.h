#ifndef DATA_H
#define DATA_H

 #include <darknet_cl/darknet.h>
#include "matrix.h"
#include "list.h"
#include "image.h"
#include "tree.h"

static inline float distance_from_edge(int x, int max)
{
    int dx = (max/2) - x;
    if (dx < 0) dx = -dx;
    dx = (max/2) + 1 - dx;
    dx *= 2;
    float dist = (float)dx/max;
    if (dist > 1) dist = 1;
    return dist;
}
void load_data_blocking(load_args args);


void print_letters(float *pred, int n);
dataDark load_data_captcha(char **paths, int n, int m, int k, int w, int h);
dataDark load_data_captcha_encode(char **paths, int n, int m, int w, int h);
dataDark load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure);
dataDark load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
dataDark load_data_super(char **paths, int n, int m, int w, int h, int scale);
dataDark load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
dataDark load_data_regression(char **paths, int n, int m, int classes, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
dataDark load_go(char *filename);


dataDark load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h);

void get_random_batch(dataDark d, int n, float *X, float *y);
dataDark get_data_part(dataDark d, int part, int total);
dataDark get_random_data(dataDark d, int num);
dataDark load_categorical_data_csv(char *filename, int target, int k);
void normalize_data_rows(dataDark d);
void scale_data_rows(dataDark d, float s);
void translate_data_rows(dataDark d, float s);
void randomize_data(dataDark d);
dataDark *split_data(dataDark d, int part, int total);
dataDark concat_datas(dataDark *d, int n);
void fill_truth(char *path, char **labels, int k, float *truth);

#endif
