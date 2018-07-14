//
// Created by zer0like on 2018/7/12.
//

#ifndef HOLE_FILLING_UTILS_H
#define HOLE_FILLING_UTILS_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <assert.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <cmath>


cv::Mat getPatch(const cv::Mat & image,const cv::Point& p,int RADIUS);
std::vector<cv::Point> FindPoint(cv::Point &p,int with, int height,int range,int RADIUS);



#endif //HOLE_FILLING_UTILS_H
