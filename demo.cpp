#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include "hole_filling.h"
#include "A6.h"

using namespace std;

int main()
{

    A6 A6("../BookArrival_Cam08.mkv","../BookArrival_Cam08_Depth.mkv");

	//algorithm test;

	clock_t start = clock();

    cv::VideoCapture cap_texture;
    cv::VideoCapture cap_depth;

    cap_texture.open("../BookArrival_Cam08.mkv");
    cap_depth.open("../BookArrival_Cam08_Depth.mkv");

    //read every frame and then change warping and get the D and I sprite
    int frame_num = cap_texture.get(cv::CAP_PROP_FRAME_COUNT);

    cv::Mat t_frame;
    cv::Mat d_frame;
    cv::Mat I_out;

    for(int i=0;i<1;i++){

        cap_texture>>t_frame;
        cap_depth>>d_frame;
        cv::cvtColor(d_frame, d_frame, CV_RGB2GRAY);

        I_out = A6.A6Porcess(t_frame,d_frame);

    }

    cap_texture.release();
    cap_depth.release();


}
