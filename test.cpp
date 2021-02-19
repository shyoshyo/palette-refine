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
// #include "vec3.h"
// #include "nlopt.h"
// #include "nearestPoint.h"
// #include "cxxopt.h"
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
#define N_REFINE (500)
#define RATE (0.2)

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

pair<float, bool> dist(const float *p, const float *q, const float *r, const float *o, float *wpqr)
{
	float qp[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
	float rp[3] = {r[0] - p[0], r[1] - p[1], r[2] - p[2]};
	float op[3] = {o[0] - p[0], o[1] - p[1], o[2] - p[2]};

	float nx = qp[1] * rp[2] - qp[2] * rp[1];
	float ny = qp[2] * rp[0] - qp[0] * rp[2];
	float nz = qp[0] * rp[1] - qp[1] * rp[0];
	float nsqr = nx * nx + ny * ny + nz * nz;

	// printf("nxyz %f %f %f   nsqr  %f\n", nx, ny, nz, nsqr);

	float &wp=wpqr[0], &wq=wpqr[1], &wr=wpqr[2];
	wp = 0.f, wq = 0.f, wr = 0.f;
	if(nsqr > 1e-3f)
	{
		float nnorm = sqrtf(nsqr);
		nx /= nnorm;
		ny /= nnorm;
		nz /= nnorm;

		float d = op[0] * nx + op[1] * ny + op[2] * nz;
		if(d < 0) return make_pair(0.f, false);

		float t[3] = {
			o[0] - d * nx,
			o[1] - d * ny,
			o[2] - d * nz,
		};

		// printf("proj %f %f %f   \n", t[0], t[1], t[2]);

		float tp[3] = {t[0] - p[0], t[1] - p[1], t[2] - p[2]};

		float npqtx = qp[1] * tp[2] - qp[2] * tp[1];
		float npqty = qp[2] * tp[0] - qp[0] * tp[2];
		float npqtz = qp[0] * tp[1] - qp[1] * tp[0];
		float kspqt = npqtx * nx + npqty * ny + npqtz * nz;

		float nptrx = tp[1] * rp[2] - tp[2] * rp[1];
		float nptry = tp[2] * rp[0] - tp[0] * rp[2];
		float nptrz = tp[0] * rp[1] - tp[1] * rp[0];
		float ksptr = nptrx * nx + nptry * ny + nptrz * nz;

		// printf("nxyz %f %f %f  \n", nx, ny, nz);

		// printf("npqtx, npqty, npqtz %f %f %f  \n", npqtx, npqty, npqtz);
		// printf("kspqt, ksptr %f %f   \n", kspqt, ksptr);

		wq = ksptr / nnorm;
		wr = kspqt / nnorm;

		if(wq < 0.f) wq = 0.f;
		if(wq > 1.f) wq = 1.f;
		if(wr < 0.f) wr = 0.f;
		if(wr > 1.f) wr = 1.f;

		wp = 1.f - wq - wr;

		// printf("wq, wr %f %f %f   \n", wq, wr, 0.f);
	}
	else
	{
		float d = op[0] * nx + op[1] * ny + op[2] * nz;
		if(d < 0) return make_pair(0.f, false);

		wp = 1.f;
		wq = 0.f;
		wr = 0.f;
	}

	float to[3] = {
		p[0] * wp + q[0] * wq + r[0] * wr - o[0],
		p[1] * wp + q[1] * wq + r[1] * wr - o[1],
		p[2] * wp + q[2] * wq + r[2] * wr - o[2],
	};
	// return make_pair(sqrtf(to[0] * to[0] + to[1] * to[1] + to[2] * to[2]), true);
	return make_pair((to[0] * to[0] + to[1] * to[1] + to[2] * to[2]), true);
}

void ddist(const float *p, const float *q, const float *r, const float *o, float *wpqr, float *dp, float *dq, float *dr, float w)
{
	const float &wp=wpqr[0], &wq=wpqr[1], &wr=wpqr[2];

	float to[3] = {
		p[0] * wp + q[0] * wq + r[0] * wr - o[0],
		p[1] * wp + q[1] * wq + r[1] * wr - o[1],
		p[2] * wp + q[2] * wq + r[2] * wr - o[2],
	};

	// float k = w / (to[0] * to[0] + to[1] * to[1] + to[2] * to[2]);
	float k = w + w;

	if(isnan(k))
	{
		printf(" start ddist k = %f\n", k);
		getchar();
	}

	if(isnan(dp[0]))
	{
		printf(" start ddist dp[0] = %f\n", dp[0]);
		getchar();
	}

	dp[0] += k * to[0] * wp;
	dp[1] += k * to[1] * wp;
	dp[2] += k * to[2] * wp;
	
	dq[0] += k * to[0] * wq;
	dq[1] += k * to[1] * wq;
	dq[2] += k * to[2] * wq;
	
	dr[0] += k * to[0] * wr;
	dr[1] += k * to[1] * wr;
	dr[2] += k * to[2] * wr;

	if(isnan(dp[0]))
	{
		printf(" end ddist dp[0] = %f,    %f %f %f\n", dp[0], k, to[0], wp);
		getchar();
	}
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


int np;
float p[10000005][3];
float center[3];

int (*f)[3] = NULL;
int nf;

int nv;


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
	float wpqr[15][3];

	double ans = 0.f;
	for(int i = 0; i < np; i++)
	{
		float mindist = 1e10f;
		// int mindistindex = 0;

		for(int j = 0; j < nf; j++)
		{
			auto d = dist(v[f[j][0]], v[f[j][1]], v[f[j][2]], p[i], wpqr[j]);
			if(!d.second) continue;


			if((d.first) > 100)
			{
				printf("loss2: d = %f\n", d.first);
				getchar();
			}

			if(mindist > d.first)
			{
				mindist = d.first;
				// mindistindex = j;
			}
		}

		if(mindist < 1e9)
		{
			ans += mindist;
		}
	}
	return ans / np;
}


void dloss2(const float (*v)[3], float (*dv)[3], float w)
{
	int count_out = 0;


	float wpqr[15][3];

	for(int i = 0; i < np; i++)
	{
		float mindist = 1e10f;
		int mindistindex = 0;

		for(int j = 0; j < nf; j++)
		{
			auto d = dist(v[f[j][0]], v[f[j][1]], v[f[j][2]], p[i], wpqr[j]);
			if(!d.second) continue;

			if((d.first) > 100)
			{
				printf("d = %f\n", d.first);
				getchar();
			}

			if(mindist > d.first)
			{
				mindist = d.first;
				mindistindex = j;
			}
		}

		if(mindist < 1e9)
		{
			int j = mindistindex;
			ddist(v[f[j][0]], v[f[j][1]], v[f[j][2]], p[i], wpqr[j], 
				dv[f[j][0]], dv[f[j][1]], dv[f[j][2]], w / np);

			count_out += 1;
		}
	}

	// printf("                                           dloss: %d / %d   %f\n", count_out, np, count_out / (float)np);
}

#define MARCH_STEP (32)
float w_march[MARCH_STEP][MARCH_STEP][MARCH_STEP];

void init_fastloss2()
{

	for(int i = 0; i < np; i++)
	{
		int x = int(p[i][0] * (MARCH_STEP - 1));
		int y = int(p[i][1] * (MARCH_STEP - 1));
		int z = int(p[i][2] * (MARCH_STEP - 1));

		if(x < 0) x = 0; if(x > (MARCH_STEP - 2)) x = (MARCH_STEP - 2);
		if(y < 0) y = 0; if(y > (MARCH_STEP - 2)) y = (MARCH_STEP - 2);
		if(z < 0) z = 0; if(z > (MARCH_STEP - 2)) z = (MARCH_STEP - 2);

		float dx = (p[i][0] * (MARCH_STEP - 1)) - x;
		float dy = (p[i][1] * (MARCH_STEP - 1)) - y;
		float dz = (p[i][2] * (MARCH_STEP - 1)) - z;

		float dx_ = 1 - dx;
		float dy_ = 1 - dy;
		float dz_ = 1 - dz;

		// printf("init_fastloss2  %f %f %f   p[i] =  %f %f %f\n", dx, dy, dz, p[i][0], p[i][1], p[i][2]);
		// getchar();

		assert(-1e-2f <= dx && dx <= 1.f + 1e-2f);
		assert(-1e-2f <= dy && dy <= 1.f + 1e-2f);
		assert(-1e-2f <= dz && dz <= 1.f + 1e-2f);

		w_march[x][y][z] += (dx_) * (dy_) * (dz_);
		w_march[x + 1][y][z] += (dx) * (dy_) * (dz_);
		w_march[x][y + 1][z] += (dx_) * (dy) * (dz_);
		w_march[x + 1][y + 1][z] += (dx) * (dy) * (dz_);
		w_march[x][y][z + 1] += (dx_) * (dy_) * (dz);
		w_march[x + 1][y][z + 1] += (dx) * (dy_) * (dz);
		w_march[x][y + 1][z + 1] += (dx_) * (dy) * (dz);
		w_march[x + 1][y + 1][z + 1] += (dx) * (dy) * (dz);
	}
}

float fastloss2(const float (*v)[3])
{
	float wpqr[15][3];

	double ans = 0.f;

	for(int i = 0; i < MARCH_STEP; i++)
		for(int j = 0; j < MARCH_STEP; j++)
			for(int k = 0; k < MARCH_STEP; k++)
			{
				if(w_march[i][j][k] <= 0) continue;

				
				float myp[3] =
				{
					i / (float)(MARCH_STEP - 1),
					j / (float)(MARCH_STEP - 1),
					k / (float)(MARCH_STEP - 1)
				}; 

				float mindist = 1e10f;
				// int mindistindex = 0;

				for(int s = 0; s < nf; s++)
				{
					auto d = dist(v[f[s][0]], v[f[s][1]], v[f[s][2]], myp, wpqr[s]);
					if(!d.second) continue;


					if((d.first) > 100)
					{
						printf("d = %f\n", d.first);
						getchar();
					}

					if(mindist > d.first)
					{
						mindist = d.first;
						// mindistindex = s;
					}
				}

				if(mindist < 1e9)
				{
					ans += mindist * w_march[i][j][k];
				}
			}
	return ans / np;
}


void dfastloss2(const float (*v)[3], float (*dv)[3], float w)
{
	float wpqr[15][3];

	for(int i = 0; i < MARCH_STEP; i++)
		for(int j = 0; j < MARCH_STEP; j++)
			for(int k = 0; k < MARCH_STEP; k++)
			{
				if(w_march[i][j][k] <= 0) continue;

				
				float myp[3] =
				{
					i / (float)(MARCH_STEP - 1),
					j / (float)(MARCH_STEP - 1),
					k / (float)(MARCH_STEP - 1)
				}; 

				float mindist = 1e10f;
				int mindistindex = 0;

				for(int s = 0; s < nf; s++)
				{
					auto d = dist(v[f[s][0]], v[f[s][1]], v[f[s][2]], myp, wpqr[s]);
					if(!d.second) continue;


					if((d.first) > 100)
					{
						printf("d = %f\n", d.first);
						getchar();
					}

					if(mindist > d.first)
					{
						mindist = d.first;
						mindistindex = s;
					}
				}

				if(mindist < 1e9)
				{
					int &s = mindistindex;
					// ans += mindist * w_march[i][j][k];
					ddist(v[f[s][0]], v[f[s][1]], v[f[s][2]], myp, wpqr[s],
						dv[f[s][0]], dv[f[s][1]], dv[f[s][2]], w * w_march[i][j][k] / np);
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


bool choose[10000005];
vector<int> chooselist;

float v[15][3];

void test()
{
	float P[3] = {5, 4, 2};
	float Q[3] = {8, 2, 9};
	float R[3] = {4, -2, 7};

	float O[3] = {6, 2, 3};

	float tmp[3];

	printf("%f\n", dist(P, Q, R, O, tmp).first);
	printf("%f\n", dist(P, R, Q, O, tmp).first);
}

int main(int argc, char** argv)
{
	/*
	test();
	return 0;
	*/

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

	for(int _ = 1; _ < K; _++)
	{
		int thischoose = 0;
		float maxdist = 0.f;

		for(int i = 1; i < np; i++)
		{
			float d = 1e10f;

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


	init_fastloss2();

	// printf("nf = %d, sizeof(f) = %lld\n", nf, sizeof(f));
	printf(" before refine: %lf %lf %lf\n", loss1(v), loss2(v), loss3(v));

	float w1 = 0.01f;
	float w2 = 10.f;
	float w3 = 1.f;
	for(int _ = 0; _ < N_REFINE; _++)
	{
		float dv[15][3];

		/*
		memset(dv, 0, sizeof dv);
		// dloss1(v, dv, w1);
		dloss2(v, dv, w2);

		for(int i = 0; i < nf; i++)
			printf(" dloss2 %f %f %f\n", dv[i][0], dv[i][1], dv[i][2]);

		memset(dv, 0, sizeof dv);
		dfastloss2(v, dv, w2);

		for(int i = 0; i < nf; i++)
			printf(" dfastloss2 %f %f %f\n", dv[i][0], dv[i][1], dv[i][2]);

		break;
		// dloss3(v, dv, w3);
		*/

		memset(dv, 0, sizeof dv);
		dloss1(v, dv, w1);
		dfastloss2(v, dv, w2);


		for(int j = 0; j < nv; j++)
		{
			for(int k = 0; k < 3; k++)
			{
				v[j][k] -= dv[j][k] * (RATE);

				// printf("%f  ", dv[j][k]);
			}
			// puts("");
		}
		// if(_ % 1 == 0) printf(" after %d: %lf %lf %lf     fastloss2 %f\n", _, loss1(v), loss2(v), loss3(v), fastloss2(v));
		if(_ % 10 == 0) printf(" after %d: %lf ?? %lf     fastloss2 %f\n", _, loss1(v), loss3(v), fastloss2(v));
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
