#include <iostream>
#include <cmath>
#include <opencv2\opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include "hole_filling.h"
#include "utils.h"

using namespace std;

int main()
{
	element aaaa;
	int tv_height, tv_width, dv_height, dv_width;
	int tv_frame, dv_frame;
	aaaa.readVideo("balloons_x264_qp14.mkv", "depth_balloons_x264_qp34.mkv");
	aaaa.processVideo(50, A1); //25, 50
	aaaa.writeVideo("balloons_x264_qp14_50.avi", "depth_balloons_x264_qp34_50.avi");
	
	system("pause");
}