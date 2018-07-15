#pragma once

#ifndef HOLE_FILLING_UTILS_H
#define HOLE_FILLING_UTILS_H

#include <opencv2\opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <assert.h>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include "hole_filling.h"
#include <cmath>

typedef std::vector<cv::Mat> video;

enum{A1, A2, A3, A4, A5, A6, A7};

class element
{
public:
	void readVideo(std::string texture_filename, std::string depth_filename);
	void writeVideo(std::string texture_filename, std::string depth_filename);
	void processVideo(const int distance, const int type);
	const int getFrame() { return frame; }
	cv::Mat getPatch(const cv::Mat & image,const cv::Point& p,int radius);
	std::vector<cv::Point> FindPoint(cv::Point &p,int with, int height,int range,int radius);

private:
	video texture_video, depth_video;
	int rows, cols, frame;
	void showVideo();
};

#endif //HOLE_FILLING_UTILS_H
