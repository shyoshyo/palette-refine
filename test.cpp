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

#define CONVHULL_3D_USE_FLOAT_PRECISION /* (optional) */
#define CONVHULL_3D_ENABLE
#include "convhull_3d.h"

// #define K (10)
#define K (5)
#define N_REFINE (30)
#define RATE (0.1)


int np;
float p[10000005][3];
float center[3];


bool choose[10000005];
vector<int> chooselist;

float v[15][3];
int nv;

int (*f)[3] = NULL;
int nf;

float dist(float *u, float *v)
{
	float x = u[0] - v[0];
	float y = u[1] - v[1];
	float z = u[2] - v[2];
	float sqr = x * x + y * y + z * z;
	if(sqr < 0) sqr = 0;
	return sqrt(sqr);
}

float det(const float *p, const float *q, const float *r)
{
	float ans = 0.f;
	ans += p[0] * q[1] * r[2] - p[0] * q[2] * r[1];
	ans += p[1] * q[2] * r[0] - p[1] * q[0] * r[2];
	ans += p[2] * q[0] * r[1] - p[2] * q[1] * r[0];
	return ans;
}

void ddet(const float *p, const float *q, const float *r, float *dp, float *dq, float *dr, float w)
{
	dp[0] += w * (q[1] * r[2] - q[2] * r[1]);
	dp[1] += w * (q[2] * r[0] - q[0] * r[2]);
	dp[2] += w * (q[0] * r[1] - q[1] * r[0]);

	dq[0] += w * (p[2] * r[1] - p[1] * r[2]);
	dq[1] += w * (p[0] * r[2] - p[2] * r[0]);
	dq[2] += w * (p[1] * r[0] - p[0] * r[1]);

	dr[0] += w * (p[1] * q[2] - p[2] * q[1]);
	dr[1] += w * (p[2] * q[0] - p[0] * q[2]);
	dr[2] += w * (p[0] * q[1] - p[1] * q[0]);
}

float det(const float *p, const float *q, const float *r, const float *o)
{
	float ans = 0.f;
	ans += (p[0] - o[0]) * (q[1] - o[1]) * (r[2] - o[2]) - (p[0] - o[0]) * (q[2] - o[2]) * (r[1] - o[1]);
	ans += (p[1] - o[1]) * (q[2] - o[2]) * (r[0] - o[0]) - (p[1] - o[1]) * (q[0] - o[0]) * (r[2] - o[2]);
	ans += (p[2] - o[2]) * (q[0] - o[0]) * (r[1] - o[1]) - (p[2] - o[2]) * (q[1] - o[1]) * (r[0] - o[0]);
	return ans;
}

void ddet(const float *p, const float *q, const float *r, const float *o, float *dp, float *dq, float *dr, float w)
{
	dp[0] += w * ((q[1] - o[1]) * (r[2] - o[2]) - (q[2] - o[2]) * (r[1] - o[1]));
	dp[1] += w * ((q[2] - o[2]) * (r[0] - o[0]) - (q[0] - o[0]) * (r[2] - o[2]));
	dp[2] += w * ((q[0] - o[0]) * (r[1] - o[1]) - (q[1] - o[1]) * (r[0] - o[0]));

	dq[0] += w * ((p[2] - o[2]) * (r[1] - o[1]) - (p[1] - o[1]) * (r[2] - o[2]));
	dq[1] += w * ((p[0] - o[0]) * (r[2] - o[2]) - (p[2] - o[2]) * (r[0] - o[0]));
	dq[2] += w * ((p[1] - o[1]) * (r[0] - o[0]) - (p[0] - o[0]) * (r[1] - o[1]));

	dr[0] += w * ((p[1] - o[1]) * (q[2] - o[2]) - (p[2] - o[2]) * (q[1] - o[1]));
	dr[1] += w * ((p[2] - o[2]) * (q[0] - o[0]) - (p[0] - o[0]) * (q[2] - o[2]));
	dr[2] += w * ((p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0]));
}

float loss1(const float (*v)[3])
{
	float ans = 0.f;
	for(int i = 0; i < nf; i++)
	{
		ans += det(v[f[i][0]], v[f[i][1]], v[f[i][2]]);
	}
	return ans;
}

void dloss1(const float (*v)[3], float (*dv)[3], float w)
{
	for(int i = 0; i < nf; i++)
	{
		ddet(v[f[i][0]], v[f[i][1]], v[f[i][2]], dv[f[i][0]], dv[f[i][1]], dv[f[i][2]], w);
	}
}


float loss2(const float (*v)[3])
{
	float ans = 0.f;
	for(int i = 0; i < nf; i++)
		for(int j = 0; j < np; j++)
		{
			float d = det(v[f[i][0]], v[f[i][1]], v[f[i][2]], p[j]);
			if(d > 0) ans += d;
		}
	return ans;
}

void dloss2(const float (*v)[3], float (*dv)[3], float w)
{
	for(int i = 0; i < nf; i++)
		for(int j = 0; j < np; j++)
		{
			float d = det(v[f[i][0]], v[f[i][1]], v[f[i][2]], p[j]);
			if(d > 0)
			{
				ddet(v[f[i][0]], v[f[i][1]], v[f[i][2]], p[j],
					dv[f[i][0]], dv[f[i][1]], dv[f[i][2]], w);
			}
		}
}

float loss3(const float (*v)[3])
{
	float ans = 0.f;
	for(int i = 0; i < nf; i++)
		for(int j = 0; j < nv; j++)
		{
			if(j == f[i][0] || j == f[i][1] || j == f[i][2]) continue;
			float d = -det(v[f[i][0]], v[f[i][1]], v[f[i][2]], v[j]);
			if(d > 0) ans += d;
		}
	return ans;
}

void dloss3(const float (*v)[3], float (*dv)[3], float w)
{
	for(int i = 0; i < nf; i++)
		for(int j = 0; j < nv; j++)
		{
			if(j == f[i][0] || j == f[i][1] || j == f[i][2]) continue;
			float d = -det(v[f[i][0]], v[f[i][1]], v[f[i][2]], v[j]);
			if(d > 0)
			{
				ddet(v[f[i][0]], v[f[i][1]], v[f[i][2]], v[j],
					dv[f[i][0]], dv[f[i][1]], dv[f[i][2]], -w);
			}
		}
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
			
			center[0] += p[np][0];
			center[1] += p[np][1];
			center[2] += p[np][2];
			
			++np;
		}

	center[0] /= np;
	center[1] /= np;
	center[2] /= np;

	{
		memset(choose, 0, sizeof choose);

		int startchoose = 0;
		float startdist = 0.f;
		
		for(int i = 1; i < np; i++)
		{
			float d = dist(center, p[i]);
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
				float _d = dist(p[i], p[j]);
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


	ch_vertex _v[K];

	for(auto i : chooselist)
	{

		_v[nv].x = v[nv][0] = p[i][0] + 0.f;
		_v[nv].y = v[nv][1] = p[i][1] + 0.f;
		_v[nv].z = v[nv][2] = p[i][2] + 0.f;

		++nv;
	}

	convhull_3d_build(_v, nv, (int**)(void*)&f, &nf);


	// printf("nf = %d, sizeof(f) = %lld\n", nf, sizeof(f));
	printf(" before refine: %lf %lf %lf\n", loss1(v), loss2(v), loss3(v));

	float w1 = 0.0001f;
	float w2 = 0.4f / np;
	float w3 = 1.f;
	for(int _ = 0; _ < N_REFINE; _++)
	{
		float dv[15][3];

		memset(dv, 0, sizeof dv);
		dloss1(v, dv, w1);
		dloss2(v, dv, w2);
		dloss3(v, dv, w3);

		for(int j = 0; j < nv; j++)
		{
			for(int k = 0; k < 3; k++)
			{
				v[j][k] -= dv[j][k] * (RATE);

				// printf("%f  ", dv[j][k]);
			}
			// puts("");
		}
		if(_ % 1 == 0) printf(" after %d: %lf %lf %lf\n", _, loss1(v), loss2(v), loss3(v));
	}


	FILE *fp = fopen(argv[2], "w");
	for(int i = 0; i < nv; i++)
	{
		fprintf(fp, "v %f %f %f\n", v[i][0], v[i][1], v[i][2]);
	}
	for(int i = 0; i < nf; i++)
	{
		fprintf(fp, "f %d %d %d\n", f[i][0], f[i][1], f[i][2]);
	}
	fclose(fp);
}
