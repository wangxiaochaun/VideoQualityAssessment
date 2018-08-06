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
const std::string path = "QP_pairs\\";

bool process(int fileindex, int qpindex, int qp_t, int qp_d, int warpingscalar, int fillmetod) {
	filename file;
	std::string input_d = path + file.findqpfileName(qpindex, 0, qp_t, fileindex);
	std::string input_t = path + file.findqpfileName(qpindex, 1, qp_d, fileindex);
	std::string outvideo_t = path + file.getFuLLName(fileindex, qpindex, qp_t, qp_d, warpingscalar, fillmetod);
	std::string outvideo_d = path +"depth_"+ file.getFuLLName(fileindex, qpindex, qp_t, qp_d, warpingscalar, fillmetod);

	element ele;

	ele.readVideo(input_t, input_d);
	if (fillmetod == 5) {
		ele.processVideo(warpingscalar, input_t, input_d);
		ele.writeVideo(outvideo_t);
	}
	else {
		ele.processVideo(warpingscalar, fillmetod);
		ele.writeVideo(outvideo_t, outvideo_d);
	}
	

	return true;
}

bool unionporcess(int fileindex) {
	int qpindex[2] = { 264, 265 };
	int qplevel[2] = { 14,34 };
	int warpingscalar[4] = { -40,-20,20,40 };
	int fillmethod[2] = {1, 4};
	int percentage = 1;
	for (int qp = 0; qp < 2; qp++) {
		//need to generate the pairs of depth and texture qp
		for (int qp_t = 0; qp_t < 2; qp_t++) {
			for (int qp_d = 0; qp_d < 2; qp_d++) {
				for (int w = 0; w < 4; w++) {
					for (int fill = 0; fill < 2; fill++) {
						process(fileindex, qpindex[qp], qplevel[qp_t], qplevel[qp_d], warpingscalar[w], fillmethod[fill]);
					}
					std::cout << "Processing percentage: " << (double)(percentage / 32.0) * 100 << "%." << std::endl;
					percentage++;
				}
			}
		}
	}
	return true;
}

int main(int argc, char ** argv)
{
	std::cout << "---------------------------------" << std::endl;
	std::cout << "| VRTS Synthesized Video Dataset|" << std::endl;
	std::cout << "|       Processing start!       |" << std::endl;
	std::cout << "---------------------------------" << std::endl;
	// 每个机器处理不同的条目
	unionporcess(1);

}