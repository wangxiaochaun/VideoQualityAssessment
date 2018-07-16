#include <iostream>
#include <cmath>
#include <opencv2\opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include "hole_filling.h"
#include "utils.h"
#include "filename.h"

using namespace std;
const std::string path = "J:\\SubjectDataSet\\QP_pairs\\";

bool process(int fileindex,int qpindex,int qp_t,int qp_d,int warpingscalar,int fillmetod) {
	filename file;
	std::string input_d = path+file.findqpfileName(qpindex, 0, qp_t, fileindex);
	std::string input_t = path+file.findqpfileName(qpindex, 1, qp_d, fileindex);
	std::string outvideo_t = path+file.getFuLLName(fileindex, qpindex, qp_t, qp_d, warpingscalar, fillmetod);

	element ele;

	ele.readVideo(input_t, input_d);
	if (fillmetod == 5) ele.processVideo(warpingscalar, input_t, input_d);
	else ele.processVideo(warpingscalar, fillmetod);
	ele.writeVideo(outvideo_t);

	return true;
}

bool unionporcess(int fileindex) {
	int qpindex[2] = { 264, 265 };
	int qplevel[2] = { 14,34 };
	int warpingscalar[4] = { -40,-20,20,40 };
	int fillmethod[3] = { 0,1,2 };
	for (int qp = 0; qp < 2; qp++) {
		//need to generate the pairs of depth and texture qp
		for (int qp_t = 0; qp_t < 2; qp_t++) {
			for (int qp_d = 0; qp_d < 2; qp_d++) {
				for (int w = 0; w < 4; w++) {
					for (int fill = 0; fill < 3; fill++) {
						process(fileindex, qpindex[qp], qplevel[qp_t], qplevel[qp_d], warpingscalar[w], fillmethod[fill]);
					}
				}
			}
		}
	}
	return true;
}

int main()
{
	//element aaaa;
	//int tv_height, tv_width, dv_height, dv_width;
	//int tv_frame, dv_frame;
	//aaaa.readVideo("balloons_x264_qp14.mkv", "depth_balloons_x264_qp34.mkv");
	//aaaa.processVideo(50, A1); //25, 50
	//aaaa.writeVideo("balloons_x264_qp14_50.avi", "depth_balloons_x264_qp34_50.avi");
	//
	//system("pause");

	filename qpfile;
	int qpindex = 264;
	int imagetype = 0;
	int qp_t= 34;
	int qp_d = 14;
	int fileindex = 1;
	int warpingscalar = -50;
	int fillmethod = 1;
	/*std::string file = qpfile.findqpfileName(qpindex, imagetype, qplevel, fileindex);
	std::string out = qpfile.getFuLLName(fileindex, qpindex, qplevel, qplevel2, warpingscalar, fillmethod);
	std::cout << out << std::endl;
	system("pause");*/

	process(fileindex, qpindex, qp_t, qp_d, warpingscalar, 5);
}