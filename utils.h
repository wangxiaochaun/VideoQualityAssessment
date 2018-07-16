#pragma once
#include <opencv2\opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include "hole_filling.h"
#include <cmath>
#include "kooper.h"

typedef std::vector<cv::Mat> video;

enum { A1, A2, A3, A4, A5, A6, A7 };

class element
{
public:
	void readVideo(std::string texture_filename, std::string depth_filename);
	void writeVideo(std::string texture_filename, std::string depth_filename);
	void processVideo(const int distance, const int type);
	void processVideo(const int distance, std::string t_file, std::string d_file);
	const int getFrame() { return frame; }

private:
	video texture_video, depth_video;
	int rows, cols, frame;
	void showVideo();
	void showAlgorithmType(int type);
};

