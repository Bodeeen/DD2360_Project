#include <glm/vec3.hpp>

#include "common.hpp"
#include "util.hpp"

#include "glad/glad.h"

#include "glm/gtc/type_ptr.hpp"

#include <chrono>
#include <random>

#include "planet.hpp"

using namespace std;
using namespace glm;
using namespace agp;

GLuint g_default_vao = 0;
glm::vec3 camera;
unsigned int shaderProgram = 0;
std::vector<glm::vec3> silicateParticles;
std::vector<glm::vec3> ironParticles;

Planet planets[2];

void init()
{
    planets[0].setOrigin(2.0, 2.0, 2.0);
    planets[1].setOrigin(-2.0, -2.0, -2.0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    for(int j=0; j<2; j++) {
        for(int i=0; i<NUM_SILICATE_PARTICLES; i++) {
            double theta = 2 * M_PI * uniform01(generator);
            double phi = acos(1 - 2 * uniform01(generator));
            double x = 2 * sin(phi) * cos(theta);
            double y = 2 * sin(phi) * sin(theta);
            double z = 2 * cos(phi);
            planets[j].addSilicateParticle(x, y, z);
        }

        for(int i=0; i<NUM_IRON_PARTICLES; i++) {
            double theta = 2 * M_PI * uniform01(generator);
            double phi = acos(1 - 2 * uniform01(generator));
            double x = 1.7 * sin(phi) * cos(theta);
            double y = 1.7 * sin(phi) * sin(theta);
            double z = 1.7 * cos(phi);
            planets[j].addIronParticle(x, y, z);
        }
    }

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

    camera = glm::vec3(0.0, 0.0, 100.0);

    glm::mat4 model, view, projection;
    model = glm::mat4(1.0f);
    view = glm::lookAt(camera, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    projection = glm::perspective(glm::radians(45.0f), (float)1280 / (float)720, 0.1f, 100.0f);

    unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
    unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
    unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");

    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
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

    //printf("FreeGLUT triggered the display() callback!\n");

    // Your rendering code must be here! Do not forget to swap the
    // front and back buffers, and to force a redisplay to keep the
    // render loop running. This functionality is available withinglm::vec3
    // FreeGLUT as well, check the assignment for more information.
    for(int i=0; i<2; i++) {
        GLint color_location = glGetUniformLocation(shaderProgram, "myColor");
        float *colorSilicate = planets[i].getSilicateColor();
        glUniform4fv(color_location, 1, colorSilicate);

        for(auto &particle : planets[i].getSilicateParticles()) {
            glm::mat4 mod(1.0f);
            glm::vec4 pat(particle, 1.0);
            glm::mat4 model(1.0f);
            model = glm::translate(model, particle);
            unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glutSolidSphere(0.02,10,10);

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glutSolidSphere(0.02,10,10);
        }

        color_location = glGetUniformLocation(shaderProgram, "myColor");
        float *colorIron = planets[i].getIronColor();
        glUniform4fv(color_location, 1, colorIron);

        for(auto &particle : planets[i].getIronParticles()) {
            glm::mat4 mod(1.0f);
            glm::vec4 pat(particle, 1.0);
            glm::mat4 model(1.0f);
            model = glm::translate(model, particle);
            unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glutSolidSphere(0.02,10,10);

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glutSolidSphere(0.02,10,10);
        }
    }

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
