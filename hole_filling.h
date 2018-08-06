#pragma once

#include <iostream>
#include <opencv2\opencv.hpp>


// Patch raduius
#define RADIUS 50
// The maximum number of pixels aroud a specified point on the target outline
#define BORDER_RADIUS 4


class algorithm
{
public:
	algorithm() { distance = 0; }
	algorithm(int distance) { this->distance = distance; }

	typedef std::vector<std::vector<cv::Point>> contours_t;
	typedef std::vector<cv::Vec4i> hierarchy_t;
	typedef std::vector<cv::Point> contour_t;

	int mod(int a, int b);

	void loadInpaintingImages(cv::Mat & colorMat, cv::Mat & maskMat, cv::Mat & grayMat);

	void showMat(const cv::String & winname, const cv::Mat & mat, int time = 5);

	void getContours(const cv::Mat & mask, contours_t & contours, hierarchy_t & hierarchy);

	double computeConfidence(const cv::Mat & confidencePatch);

	cv::Mat getPatch(const cv::Mat & image, const cv::Point & p);

	void getDerivatives(const cv::Mat & grayMat, cv::Mat & dx, cv::Mat & dy);

	cv::Point2f getNormal(const contour_t & contour, const cv::Point & point);

	void computePriority(const contours_t & contours, const cv::Mat & grayMat, const cv::Mat & confidenceMat, cv::Mat & priorityMat);

	void transferPatch(const cv::Point & psiHatQ, const cv::Point & psiHatP, cv::Mat & mat, const cv::Mat & maskMat);

	cv::Mat computeSSD(const cv::Mat & tmplate, const cv::Mat & source, const cv::Mat & tmplateMask);

	bool Fehn_interpolation(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn);	//A1

	bool Fehn_inpainting(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn);		//A2

	bool Tanimoto(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn);				//A3

	bool Muller(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn);				//A4

	bool Ndijiki(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn);				//A5

	bool warping_1d(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn);

	const cv::Mat mask_detection(const cv::Mat & input);

	bool post_processing(cv::Mat & I_syn, int distance);

private:
	int distance;
	//int direction;

	void setDistance(const int distance) { this->distance = distance; }
	//void setDirection(const int direction) { this->direction = direction; }
public:
	const int getDistance() { return this->distance; }
	//const int getDirection() { return this->direction; }
};