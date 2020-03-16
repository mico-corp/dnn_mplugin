#ifndef LIST_H
#define LIST_H
 #include <darknet_cl/darknet.h>

listDark *make_list();
int list_find(listDark *l, char *val);

void list_insert(listDark *, char *);


void free_list_contents(listDark *l);

#endif
