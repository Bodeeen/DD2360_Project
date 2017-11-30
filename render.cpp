#include "simulation.h"
#include "helper.h"
#include "common.hpp"
#include "util.hpp"

using namespace std;
using namespace glm;
using namespace agp;

#define NUM_PARTICLES 1000
#define TIME_STEP 1

GLuint g_default_vao = 0;
GLuint program;
mat4 M;
mat4 V;
mat4 P;
mat4 MVP;
GLuint UniLoc;

int window_width = 1280;
int window_height = 800;

void init()
{
    // Generate and bind the default VAO
    glGenVertexArrays(1, &g_default_vao);
    glBindVertexArray(g_default_vao);

    // Set the background color (RGBA)
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    program = util::loadShaders("vert_shader.glsl", "frag_shader.glsl");
    glUseProgram(program);

    camPos = vec3(0,0,50);
    FOV = 45.0f;

    V = lookAt(
               camPos, // camera position
               vec3(0.0), // look at origin
               vec3(0.f, 1.f, 0.f));  // Head is up

    P = perspective(radians(FOV), (GLfloat) window_width / (GLfloat) window_height, 0.1f, 100.0f);

    MVP = P*V*M;
    UniLoc = glGetUniformLocation(program, "MVP");
    glUniformMatrix4fv(UniLoc, 1, GL_FALSE, value_ptr(MVP));
    // Your OpenGL settings, such as alpha, depth and others, should be
    // defined here! For the assignment, we only ask you to enable the
    // alpha channel.
}

void release()
{
    // Release the default VAO
    glDeleteVertexArrays(1, &g_default_vao);

    // Do not forget to release any memory allocation here!
}

void display()
{
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    printf("FreeGLUT triggered the display() callback!\n");

    // Your rendering code must be here! Do not forget to swap the
    // front and back buffers, and to force a redisplay to keep the
    // render loop running. This functionality is available within
    // FreeGLUT as well, check the assignment for more information.

    // Important note: The following function flushes the rendering
    // queue, but this is only for single-buffered rendering. You
    // must replace this function following the previous indications.
    glutSwapBuffers();
    glutPostRedisplay();
}

int start_render(int argc, char **argv)
{
  // Initialize FreeGLUT and create the window
  glutInit(&argc, argv);

  // Setup the window (e.g., size, display mode and so on)
  glutInitWindowSize(window_width, window_height);
  //glutInitWindowPosition( ... );
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

  // Make FreeGLUT return from the main rendering loop
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                GLUT_ACTION_GLUTMAINLOOP_RETURNS);

  // Create the window and associate the callbacksglutKeyboardFunc(KeyboardCallback);
  glutCreateWindow("Applied GPU Programming");
  glutDisplayFunc(display);
  // glutIdleFunc( ... );
  glutReshapeFunc(ReshapeWindow);
  //glutKeyboardFunc(KeyboardCallback);
  //glutSpecialFunc(SpecialInput);
  //glutMouseFunc(MouseFunc);
  // glutMotionFunc( ... );

  // Init GLAD to be able to access the OpenGL API
  if (!gladLoadGL())
  {
      return GL_INVALID_OPERATION;
  }

  // Display OpenGL information
  util::displayOpenGLInfo();
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  // Initialize the 3D view
  init();

  // Launch the main loop for rendering
  glutMainLoop();

  // Release all the allocated memory
  release();
}
