#ifndef HELPER_H
#define HELPER_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h> //getopt
#include <ctype.h>

bool handle_opt( int argc,  char *argv[], int *n, int *it, int *size);

#endif