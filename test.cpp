/*
 * OGL01Shape3D.cpp: 3D Shapes
 */
#include <GL/glut.h>  // GLUT, include glu.h and gl.h
#include <cstdio>
#include <utility>
#include <cmath>
#include <vector>
#include <tuple>
#include <iostream>
#include <nlopt.h>
#include <nlopt.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <stdio.h>
#include <time.h>
#include "vec3.h"
#include "nlopt.h"
#include "nearestPoint.h"
#include "cxxopt.h"
#include <algorithm>
#include <chrono>
using namespace std;
using namespace cv;
using namespace nlopt;

#define K (10)

int np;
float p[10000005][3];
bool choose[10000005];

vector<int> chooselist;

float dist(int u, int v)
{
	float x = p[u][0] - p[v][0];
	float y = p[u][1] - p[v][1];
	float z = p[u][2] - p[v][2];
	float sqr = x * x + y * y + z * z;
	if(sqr < 0) sqr = 0;
	return sqrt(sqr);
}

int main(int argc, char** argv)
{
    Mat img = imread(argv[1]);
    for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			//std::cout << "(x, y) = " << "(" << x << ", " << y << ")" << std::endl;
			const uchar* pointer = img.ptr<uchar>(i);
			
			p[np][0] = pointer[3 * j + 2] / 255.0f;
			p[np][1] = pointer[3 * j + 1] / 255.0f;
			p[np][2] = pointer[3 * j] / 255.0f;
			++np;
		}

	{
		memset(choose, 0, sizeof choose);

		int startchoose = 0;
		float startdist = 0.f;
		
		for(int i = 1; i < np; i++)
		{
			float d = dist(0, i);
			if(startdist < d)
			{
				startdist = d;
				startchoose = i;
			}
		}

		chooselist.push_back(startchoose);
		choose[startchoose] = 1;
		printf("%d\n", startchoose);
	}

	for(int i = 1; i < K; i++)
	{
		int thischoose = 0;
		float maxdist = 0.f;

		for(int i = 1; i < np; i++)
		{
			float d = 1e60;

			for(auto j : chooselist)
			{
				float _d = dist(i, j);
				if(d > _d) d = _d;
			}
			
			if(maxdist < d)
			{
				maxdist = d;
				thischoose = i;
			}
		}

		chooselist.push_back(thischoose);
		choose[thischoose] = 1;
		printf("%d\n", thischoose);
	}

	FILE *fp = fopen(argv[2], "w");

	for(auto i : chooselist)
	{
		fprintf(fp, "v %f %f %f\n", p[i][0], p[i][1], p[i][2]);
	}





}