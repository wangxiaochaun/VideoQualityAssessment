#include "filename.h"



filename::filename()
{
}


filename::~filename()
{
}

std::string filename::getFuLLName(int fileindex,int qpindex, int qpLevel1, int qplevel2, int warpingscalear, int fillmethodindex) {

	//like this balloons_x264_qp14_50.avi
	std::string suffix = ".avi";

	std::string name = this->file[fileindex] + "_x" + this->qp[qpindex] + "_qp" + this->qpLevel[qpLevel1] + "_qp" + 
		this->qpLevel[qplevel2] + "_" + std::to_string(warpingscalear) + "_"+this->method[fillmethodindex]+suffix;

	return name;
}
//265_t34_Treeflight.mkv
std::string filename::analyseName(std::string filename) {
	return NULL;
}

std::string filename::findqpfileName(int qpindex, int imageindex, int qplevelindex, int fileindex) {
	std::string suffix = ".mkv";
	return this->qp[qpindex] + "_" + this->imagetype[imageindex] + this->qpLevel[qplevelindex] + "_" + this->file[fileindex] + suffix;
}
