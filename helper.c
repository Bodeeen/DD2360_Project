#include "helper.h"

bool handle_opt( int argc, char *argv[], int *n, int *it, int *size){
    int opt, index;
    /* get options */
    opt = getopt( argc, argv, "n:i:s:" );

    while( opt != -1 ) {
        switch(opt){
            case 'n':{
                *n = atoi(optarg);
                break;
            }
            case 'i':{
                *it = atoi(optarg);
                break;
            }
            case 's':{
                *size = atoi(optarg);
                break;
            }
            case '?':{
                if (optopt == 'n'){
                    fprintf (stderr, "Option -%c requires an argument.\n missing NUM_PARTICLES", optopt);
                }
                else if (opt == 'i')
                {
                    fprintf (stderr, "Option -%c requires an argument.\n missing NUM_ITERATIONS", optopt);
                }
                else if (opt == 's')
                {
                    fprintf (stderr, "Option -%c requires an argument.\n missing BLOCK_SIZE", optopt);
                }

                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);

                return false;
            }
            default:{
                printf("switch default, opt:%c\n", (char) opt);
                return false;
            }
        }
        /*  get next option */
        opt = getopt( argc, argv, "n:i:s:" );
    }

    for (index = optind; index < argc; index++)
    {
        printf("Non-option argument %s\n", argv[index]);
    }
        
    if (*n==0)
    {
        *n = 100000;
        printf("Default NUM_PARTICLES: %d\n", *n);
    }

    if (*it==0)
    {
        *it = 10;
        printf("Default NUM_ITERATIONS :%d\n", *it);
    }

    if (*size==0)
    {
        *size = 128;
        printf("Default BLOCK_SIZE :%d\n", *size);
    }
    return true;
}


