/*
 * OGL01Shape3D.cpp: 3D Shapes
 */
#include <GL/glut.h>  // GLUT, include glu.h and gl.h
#include <cstdio>
#include <utility>
#include <cmath>
#include <vector>
#include <tuple>
using namespace std;


/* Global variables */
char title[] = "3D Shapes";
 
/* Initialize OpenGL Graphics */
void initGL() {
   glClearColor(0.5f, 0.5f, 0.5f, 1.0f); // Set background color to black and opaque
   glClearDepth(1.0f);                   // Set background depth to farthest
   glEnable(GL_DEPTH_TEST);   // Enable depth testing for z-culling
   glDepthFunc(GL_LEQUAL);    // Set the type of depth-test
   glShadeModel(GL_SMOOTH);   // Enable smooth shading
   glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);  // Nice perspective corrections
}

float rot_x = 0.f;
float rot_y = 0.f;

void MyMouseDrag(int x, int y)
{
   // if(!pressed) return;

   rot_x = x;
   rot_y = -y;

   // if(rot_y > 90.f) rot_y = 90.f;
   // if(rot_y < -90.f) rot_y = -90.f;

   glutPostRedisplay();
}

int nv = 0;
int nf = 0;

float v[10005][5];
int f[10005][5];

/* Handler for window-repaint event. Called back when the window first appears and
   whenever the window needs to be re-painted. */
void display() {
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear color and depth buffers
   glMatrixMode(GL_MODELVIEW);     // To operate on model-view matrix
 
   // Render a color-cube consisting of 6 quads with different colors
   glLoadIdentity();                 // Reset the model-view matrix
   glTranslatef(0.f, 0.0f, -4.0f);  // Move right and into the screen
   glRotatef(rot_x, 0.f, 1.f, 0.f);
   glRotatef(rot_y, 0.f, 0.f, 1.f);
 
   glLineWidth(.5f);
   glBegin(GL_LINES);                // Begin drawing the color cube with 6 quads
   for(int i = 0; i <= 1; i++)
   {
      for(int j = 0; j <= 1; j++)
      {
         glColor3f(i, j, 0.0f);
         glVertex3f(i * 2 - 1, j * 2 - 1, -1.0f);
         glColor3f(i, j, 1.0f);
         glVertex3f(i * 2 - 1, j * 2 - 1, 1.0f);

         glColor3f(i, 0.0f, j);
         glVertex3f(i * 2 - 1, -1.0f, j * 2 - 1);
         glColor3f(i, 1.0f, j);
         glVertex3f(i * 2 - 1, 1.0f, j * 2 - 1);

         glColor3f(0.0f, i, j);
         glVertex3f(-1.0f, i * 2 - 1, j * 2 - 1);
         glColor3f(1.0f, i, j);
         glVertex3f(1.0f, i * 2 - 1, j * 2 - 1);
      }
   }
   glEnd();  // End of drawing color-cube
   
   glPointSize(10);
   glBegin(GL_POINTS);
   for(int i = 0; i < nv; i++)
   {
      glColor3f(v[i][0], v[i][1], v[i][2]);
      glVertex3f(v[i][0] * 2 - 1, v[i][1] * 2 - 1, v[i][2] * 2 - 1);
   }
   glEnd();
   
   glLineWidth(2.f);
   glBegin(GL_LINES);
   for(int i = 0; i < nf; i++)
   {
      int a = f[i][0];
      int b = f[i][1];
      int c = f[i][2];

      for(auto p : {make_pair(a, b), make_pair(b, c), make_pair(c, a)})
      {
         glColor3f(v[p.first][0], v[p.first][1], v[p.first][2]);
         glVertex3f(v[p.first][0] * 2 - 1, v[p.first][1] * 2 - 1, v[p.first][2] * 2 - 1);

         glColor3f(v[p.second][0], v[p.second][1], v[p.second][2]);
         glVertex3f(v[p.second][0] * 2 - 1, v[p.second][1] * 2 - 1, v[p.second][2] * 2 - 1);
      }

      float x2 = v[b][0] - v[a][0];
      float y2 = v[b][1] - v[a][1];
      float z2 = v[b][2] - v[a][2];
      float x3 = v[c][0] - v[a][0];
      float y3 = v[c][1] - v[a][1];
      float z3 = v[c][2] - v[a][2];

      float nx = y2 * z3 - z2 * y3;
      float ny = z2 * x3 - x2 * z3;
      float nz = x2 * y3 - y2 * x3;

      float nsqure = nx * nx + ny * ny + nz * nz;
      if(nsqure > 1e-6f)
      {
         float nnorm = sqrtf(nsqure);
         nx /= nnorm;
         ny /= nnorm;
         nz /= nnorm;

         for(auto w : {
            tuple<float, float, float, float>(18.f, 1.f, 1.f, 0.05f),
            tuple<float, float, float, float>(1.f, 18.f, 1.f, 0.05f),
            tuple<float, float, float, float>(1.f, 1.f, 18.f, 0.05f),
            tuple<float, float, float, float>(1.f, 1.f, 1.f, 0.3333333333333f),
         } )
         {

            float cx = (get<0>(w) * v[a][0] + get<1>(w) * v[b][0] + get<2>(w) * v[c][0]) * get<3>(w);
            float cy = (get<0>(w) * v[a][1] + get<1>(w) * v[b][1] + get<2>(w) * v[c][1]) * get<3>(w);
            float cz = (get<0>(w) * v[a][2] + get<1>(w) * v[b][2] + get<2>(w) * v[c][2]) * get<3>(w);


            glColor3f(cx, cy, cz);
            glVertex3f(cx * 2 - 1, cy * 2 - 1, cz * 2 - 1);

            cx += nx * 0.1f;
            cy += ny * 0.1f;
            cz += nz * 0.1f;

            glColor3f(cx, cy, cz);
            glVertex3f(cx * 2 - 1, cy * 2 - 1, cz * 2 - 1);
         }
      }
   }
   glEnd();

   glutSwapBuffers();  // Swap the front and back frame buffers (double buffering)
}
 
/* Handler for window re-size event. Called back when the window first appears and
   whenever the window is re-sized with its new width and height */
void reshape(GLsizei width, GLsizei height) {  // GLsizei for non-negative integer
   // Compute aspect ratio of the new window
   if (height == 0) height = 1;                // To prevent divide by 0
   GLfloat aspect = (GLfloat)width / (GLfloat)height;
 
   // Set the viewport to cover the new window
   glViewport(0, 0, width, height);
 
   // Set the aspect ratio of the clipping volume to match the viewport
   glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
   glLoadIdentity();             // Reset
   // Enable perspective projection with fovy, aspect, zNear and zFar
   gluPerspective(45.0f, aspect, 0.1f, 100.0f);
}
 
/* Main function: GLUT runs as a console application starting at main() */
int main(int argc, char** argv) {

   FILE* fp = fopen(argv[1], "r");

   for(;;)
   {
      char str[5];
      if(fscanf(fp, "%s", str) == EOF) break;

      if(str[0] == 'v')
      {
         float x, y, z;
         fscanf(fp, "%f %f %f", &x, &y, &z);
         v[nv][0] = x;
         v[nv][1] = y;
         v[nv][2] = z;
         ++nv;
      }
      else if(str[0] == 'f')
      { 

         int x, y, z;
         fscanf(fp, "%d %d %d", &x, &y, &z);
         f[nf][0] = x;
         f[nf][1] = y;
         f[nf][2] = z;
         ++nf;
      }
   }

   fclose(fp);

   printf("nv = %d, nf = %d\n", nv, nf);


   glutInit(&argc, argv);            // Initialize GLUT
   glutInitDisplayMode(GLUT_DOUBLE); // Enable double buffered mode
   glutInitWindowSize(640, 480);   // Set the window's initial width & height
   glutInitWindowPosition(50, 50); // Position the window's initial top-left corner
   glutCreateWindow(title);          // Create window with the given title
   glutDisplayFunc(display);       // Register callback handler for window re-paint event
   glutReshapeFunc(reshape);       // Register callback handler for window re-size event
   // glutMouseFunc(MyMouse);
   glutMotionFunc(MyMouseDrag);
   initGL();                       // Our own OpenGL initialization
   glutMainLoop();                 // Enter the infinite event-processing loop
   return 0;
}

