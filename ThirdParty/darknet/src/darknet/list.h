#ifndef LIST_H
#define LIST_H
#include <darknet/darknet.h>

listDark *make_list();
int list_find(listDark *l, void *val);

void list_insert(listDark *, void *);


void free_list_contents(listDark *l);

#endif
