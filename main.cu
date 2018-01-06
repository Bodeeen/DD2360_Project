#include <glm/vec3.hpp>
#include "glad/glad.h"
#include "glm/gtc/type_ptr.hpp"

#include <cuda_gl_interop.h>

#include <memory>
#include <chrono>
#include <random>

#include "planet.hpp"
#include "simulation.h"

#include "common.hpp"
#include "util.hpp"

using namespace std;
using namespace glm;
using namespace agp;

GLuint g_default_vao = 0;
glm::vec3 camera;
unsigned int shaderProgram = 0;

//std::unique_ptr<GPUSimulation> simulator;

const int num_particles = 2*(NUM_SILICATE_PARTICLES + NUM_IRON_PARTICLES);

GPUSimulation simulator;
cudaGraphicsResource_t ssbo_handle;
// This will identify our vertex buffer
GLuint vertexbuffer;
GLuint materialBuffer;
GLuint IndexVBOID;
GLuint ssbo_pos;
//Particle_vec4 part_array[num_particles];

GLfloat triangle[] = {0.0f, 150.0f, 0.0f,
                  -150.0f, -150.0f, 0.0f,
                  150.0f, -150.0f, 0.0f};

void init()
{
  //simulator = std::make_unique<GPUSimulation>();
  simulator.init();

  // Generate and bind the default VAO
  glGenVertexArrays(1, &g_default_vao);
  glBindVertexArray(g_default_vao);

  // Set the background color (RGBA)
  glClearColor(0.0f, 0.0f, 0.0f, 0.5f);

  // Your OpenGL settings, such as alpha, depth and others, should be
  // defined here! For the assignment, we only ask you to enable the
  // alpha channel.
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  shaderProgram = util::loadShaders("vt.glsl", "fg.glsl");
  glUseProgram(shaderProgram);

  camera = glm::vec3(0.0, 0.0, 50000.0);

  glm::mat4 model, view, projection;
  model = glm::mat4(1.0f);
  view = glm::lookAt(camera, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
  projection = glm::perspective(glm::radians(60.0f), (float)1280 / (float)720, 0.1f, 100000.0f);

  unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
  unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
  unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");

  glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
  glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

  glGenBuffers(1, &ssbo_pos);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_pos);
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Particle)*simulator.all.size(), &simulator.all[0], GL_DYNAMIC_DRAW);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

  cudaGraphicsGLRegisterBuffer(&ssbo_handle, ssbo_pos, cudaGraphicsRegisterFlagsNone);

  // Generate 1 buffer, put the resulting identifier in vertexbuffer
  glGenBuffers(1, &vertexbuffer);

  // The following commands will talk about our 'vertexbuffer' buffer
  glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
  // Give our vertices to OpenGL.
  glBufferData(GL_ARRAY_BUFFER, sizeof(triangle), &triangle, GL_DYNAMIC_DRAW);

  ushort pindices[3];
  pindices[0] = 0;
  pindices[1] = 1;
  pindices[2] = 2;

  glGenBuffers(1, &IndexVBOID);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexVBOID);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ushort)*3, pindices, GL_DYNAMIC_DRAW);

}

void release()
{
  // Release the default VAO
  glDeleteVertexArrays(1, &g_default_vao);

  // Do not forget to release any memory allocation here!
  simulator.release();
}

void display()
{
  simulator.update(ssbo_handle);
  // Clear the screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  //printf("FreeGLUT triggered the display() callback!\n");


  // Your rendering code must be here! Do not forget to swap the
  // front and back buffers, and to force a redisplay to keep the
  // render loop running. This functionality is available withinglm::vec3
  // FreeGLUT as well, check the assignment for more information.

  GLint color_location = glGetUniformLocation(shaderProgram, "myColor");
  glUniform4fv(color_location, 1, simulator.planets[0]->getSilicateColor());
  color_location = glGetUniformLocation(shaderProgram, "myColor");

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);

  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0); //For glDrawElements
  // Draw the triangle !
  // glDrawArrays(GL_TRIANGLES, 0, 3); // Starting from vertex 0; 3 vertices total -> 1 triangle
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo_pos);
  timeval a, b;
  gettimeofday(&a, NULL);
  glDrawElementsInstanced(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, NULL, num_particles);
  gettimeofday(&b, NULL);
  std::cout << "OpenGL Rendering: " << getElapsed(a, b) << "s" << std::endl;
  glDisableVertexAttribArray(0);

  // for(auto &particle : simulator.all) {
  //   //std::cout << particle.position.x << "," << particle.position.y << "," << particle.position.z << std::endl;
  //   /* if it is iron */
  //   glm::mat4 mod(1.0f);
  //   glm::mat4 model(1.0f);
  //   model = glm::translate(model, particle.position);
  //   if(particle.material) {
  //     glUniform4fv(color_location, 1, simulator.planets[0]->getIronColor());
  //   }
  //   else {
  //     glUniform4fv(color_location, 1, simulator.planets[0]->getSilicateColor());
  //   }
  //
  //   unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
  //   glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
  //   glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  //   glutSolidSphere(150,10,10);
  //
  //   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  //   glutSolidSphere(150,10,10);
  //
  // }
//  sleep(5);

//  for(int i=0; i<2; i++) {
//    GLint color_location = glGetUniformLocation(shaderProgram, "myColor");
//    glUniform4fv(color_location, 1, simulator.planets[0]->getSilicateColor());
//
//    for(auto &particle : simulator.planets[i]->getSilicateParticles()) {
//      std::cout << particle.position.x << "," << particle.position.y << "," << particle.position.z << std::endl;
//      glm::mat4 mod(1.0f);
//      glm::mat4 model(1.0f);
//      model = glm::translate(model, particle.position);
//      unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
//      glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
//      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
//      glutSolidSphere(150,10,10);
//
//      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//      glutSolidSphere(150,10,10);
//    }
//
//    color_location = glGetUniformLocation(shaderProgram, "myColor");
//    glUniform4fv(color_location, 1, simulator.planets[0]->getIronColor());
//
//    for(auto &particle : simulator.planets[i]->getIronParticles()) {
//      glm::mat4 mod(1.0f);
//      glm::mat4 model(1.0f);
//      model = glm::translate(model, particle.position);
//      unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
//      glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
//      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
//      glutSolidSphere(150,10,10);
//
//      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//      glutSolidSphere(150,10,10);
//    }
//  }

  // Important note: The following function flushes the rendering
  // queue, but this is only for single-buffered rendering. You
  // must replace this function following the previous indications.
  //glFlush();

  glutSwapBuffers();
  glutPostRedisplay();
}

void processKeys(unsigned char key, int x, int y) {
  printf("%d ", key);
  switch(key) {
    case 27:
    throw "exit";
    break;
    case '-': {
      camera.z--;
      glm::mat4 view = glm::lookAt(camera, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
      unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
      glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    } break;
    case '+': {
      camera.z++;
      glm::mat4 view = glm::lookAt(camera, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
      unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
      glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    } break;

  }
}

void processSpecialKeys(int key, int x, int y)
{
  switch(key) {
    case GLUT_KEY_LEFT: {
      //camera = glm::rotate(camera, glm::radians(-2.0f), glm::vec3(0.0, 1.0, 0.0));
      glm::mat4 rotationMat(1);
      rotationMat = glm::rotate(rotationMat, 1.0f, glm::vec3(0.0, 1.0, 0.0));
      //vec = glm::vec3(rotationMat * glm::vec4(vec, 1.0));
      glm::vec4 newCamera(camera.x, camera.y, camera.z, 1.0);
      newCamera = rotationMat * newCamera;
      //camera = rotationMat * camera;
      camera = newCamera;
      glm::mat4 view = glm::lookAt(camera, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
      unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
      glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    } break;
    case GLUT_KEY_RIGHT: {
      //camera = glm::rotate(camera, glm::radians(-2.0f), glm::vec3(0.0, 1.0, 0.0));
      glm::mat4 rotationMat(1);
      rotationMat = glm::rotate(rotationMat, -1.0f, glm::vec3(0.0, 1.0, 0.0));
      //vec = glm::vec3(rotationMat * glm::vec4(vec, 1.0));
      glm::vec4 newCamera(camera.x, camera.y, camera.z, 1.0);
      newCamera = rotationMat * newCamera;
      //camera = rotationMat * camera;
      camera = newCamera;
      glm::mat4 view = glm::lookAt(camera, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
      unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
      glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    } break;
  }
}

int main(int argc, char **argv)
{
  // Initialize FreeGLUT and create the window
  glutInit(&argc, argv);

  // Setup the window (e.g., size, display mode and so on)
  glutInitWindowSize(1280, 720);
  glutInitWindowPosition(100, 100);
  glutInitDisplayMode(GLUT_RGBA);

  // Make FreeGLUT return from the main rendering loop
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
    GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    // Create the window and associate the callbacks
    glutCreateWindow("Applied GPU Programming");
    glutDisplayFunc(display);
    // glutIdleFunc( ... );
    // glutReshapeFunc( ... );
    glutKeyboardFunc(processKeys);
    glutSpecialFunc(processSpecialKeys);
    // glutMouseFunc( ... );
    // glutMotionFunc( ... );

    // Init GLAD to be able to access the OpenGL API
    if (!gladLoadGL())
    {
      return GL_INVALID_OPERATION;
    }

    // Display OpenGL information
    util::displayOpenGLInfo();

    // Initialize the 3D view
    init();

    // Launch the main loop for rendering
    glutMainLoop();

    // Release all the allocated memory
    release();

    return 0;
  }
