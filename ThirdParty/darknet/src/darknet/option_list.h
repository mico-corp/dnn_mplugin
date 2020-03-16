#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;


int read_option(char *s, listDark *options);
void option_insert(listDark *l, char *key, char *val);
char *option_find(listDark *l, char *key);
float option_find_float(listDark *l, char *key, float def);
float option_find_float_quiet(listDark *l, char *key, float def);
void option_unused(listDark *l);

#endif
