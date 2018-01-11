#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <glad/glad.h>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

//#define NUM_SILICATE_PARTICLES 6000
//#define NUM_IRON_PARTICLES 9000

#define PATH_MAX    4096
#define GL_SUCCESS  0

#define Dia 376.78
#define M_Si 7.4161e+19
#define M_Fe 1.9549e+20
#define K_Si 7.2785e+10
#define K_Fe 2.9114e+11
#define KRP_Si 0.01
#define KRP_Fe 0.02
#define SDP_Si 0.001
#define SDP_Fe 0.01
#define Core_D_Si 376.403
#define Core_D_Fe 376.026
#define GMsiMfe 9.6759e+29
#define GMsiMsi 3.67065e+29
#define GMfeMfe 2.55059e+30

//#define GMsiMfe 9.6759e+19
//#define GMsiMsi 3.67065e+19
//#define GMfeMfe 2.55059e+20


#define G 6.67408e-11
#define eps 47.0975
#define time_step 5.8117

#define NUM_ITERATIONS 1000000LL
#define BLOCK_SIZE 256

#define RImpactor 6371.0f
#define RInnershell 3185.5f
#define Xcenter 3185.0f
#define Zxenter 0.0f
#define Vinitx 3.2416f

extern int NUM_SILICATE_PARTICLES;
extern int NUM_IRON_PARTICLES;
typedef uint8_t BYTE;

typedef struct{
        glm::vec4 position;
        glm::vec4 velocity;
        unsigned int material; //True is iron
        float __padding[3];
} Particle;

typedef struct{
        glm::vec4 position;
        glm::vec4 velocity;
        unsigned int material; //True is iron
        float __padding[3];
} Particle_vec4;

#endif
