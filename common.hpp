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

#define PATH_MAX    4096
#define GL_SUCCESS  0

#define D 376.78
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
#define GMM 9.6759e+29

#define G 6.67408e-11
#define eps 47.0975
#define time_step 5.8117

typedef uint8_t BYTE;
