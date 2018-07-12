#include <iostream>
#include <cmath>
#include <opencv2\opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include "hole_filling.h"

using namespace std;

int main()
{
	cv::Mat I_ref = cv::imread("texture.bmp");
	cv::Mat D_ref = cv::imread("depth.bmp");
	cv::cvtColor(D_ref, D_ref, CV_RGB2GRAY);

	cv::Mat I_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8UC3);
	cv::Mat D_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8U);

	//cv::imshow("I_ref", D_ref);

	algorithm test;

	clock_t start = clock();

	test.Ndijiki(I_ref, D_ref, I_syn, D_syn);

	clock_t ends = clock();

	std::cout << "Running time of 3D warping (ms): " << static_cast<double>(ends - start) / CLOCKS_PER_SEC * 1000 << std::endl;
	

	cv::waitKey(0);
}