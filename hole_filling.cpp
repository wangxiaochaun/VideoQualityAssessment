#include "hole_filling.h"
#include <time.h>
#include <vector>

#define COLOR_WHITE 255
#define COLOR_BLACK 0

bool algorithm::Fehn_interpolation(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn)
{
	// 这里改变了传入参考视点深度图像的类型（前提是深度图是单通道的）

	cv::GaussianBlur(D_ref, D_ref, cv::Size(3, 3), 0, 0);

	warping_1d(I_ref, D_ref, I_syn, D_syn);

	// 裁剪掉边框
	int s = 25; // 默认裁剪掉的边的宽度
	cv::Rect myROI(s, 0, I_syn.size().width - 2 * s, I_syn.size().height);

	cv::Mat mask;

	I_syn(myROI).copyTo(mask);

	cv::resize(mask, mask, I_syn.size(), 0, 0, cv::INTER_NEAREST);

	cv::medianBlur(mask, mask, 3);
	cv::imshow("mask", mask);

	mask.copyTo(I_syn);

	return true;
}

bool algorithm::Fehn_inpainting(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn)
{
	// 本处理相当耗时，大概800ms左右，

	cv::GaussianBlur(D_ref, D_ref, cv::Size(3, 3), 0, 0);

	warping_1d(I_ref, D_ref, I_syn, D_syn);

	//test
	//cv::imshow("raw", I_syn);

	cv::Mat mask = mask_detection(I_syn);

	//clock_t start = clock();

	cv::inpaint(I_syn, mask, I_syn, 3, CV_INPAINT_TELEA);

	//clock_t ends = clock();

	//std::cout << "Running time of mask detection (ms): " << static_cast<double>(ends - start) / CLOCKS_PER_SEC * 1000 << std::endl;

	//test
	//cv::imshow("inpainting", I_syn);

	return true;
}

bool algorithm::Tanimoto(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn)
{
	// VSRS算法的简化版本，目前是这么做的：首先，对warping后的深度图做后处理（双边滤波）；然后根据深度值选择背景像素修补空洞；最后再用inpainting过一遍（inpainting太过耗时，可以看情况是否采用）
	warping_1d(I_ref, D_ref, I_syn, D_syn);

	//test
	cv::imshow("before background filling", I_syn);
	//cv::imshow("D_syn", D_syn);

	cv::Mat temp = cv::Mat(D_syn.size(), CV_8U);

	// 第三个参数是滤波器大小，过大的滤波器执行效率低，这里取9*9， d=0时，滤波器大小由后面两个高斯核sigma决定；第四、第五是值域和空域滤波核的sigma，一般取大于10
	cv::bilateralFilter(D_syn, temp, 9, 25, 25);

	//test
	//cv::imshow("After bilateral filter", temp);

	cv::Size size = I_syn.size();
	int threshold = 10; //深度差阈值，可调

	for (int i = 0; i < size.height; i++)
		for (int j = 0; j < size.width; j++)
		{
			if (I_syn.at<cv::Vec3b>(i, j)[0] == COLOR_BLACK && I_syn.at<cv::Vec3b>(i, j)[1] == COLOR_BLACK && I_syn.at<cv::Vec3b>(i, j)[2] == COLOR_BLACK)
			{
				if (D_syn.at<uchar>(i, j + 3) - D_syn.at<uchar>(i, j - 3) > threshold)
					I_syn.at<cv::Vec3b>(i, j) = I_syn.at<cv::Vec3b>(i, j - 3);
				else if (D_syn.at<uchar>(i, j - 3) - D_syn.at<uchar>(i, j + 3) > threshold)
					I_syn.at<cv::Vec3b>(i, j) = I_syn.at<cv::Vec3b>(i, j + 3);
			}
		}

	//test
	//cv::imshow("after backgroud filling", I_syn);

	cv::Mat mask = mask_detection(I_syn);
	cv::inpaint(I_syn, mask, I_syn, 3, CV_INPAINT_TELEA);
	//test
	cv::imshow("after backgroud inpainting", I_syn);

	return true;
}

bool algorithm::Muller(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn)
{
	// Muller方法和VSRS一样，依赖于两视点。这里我们只能做简化。其后处理部分和VSRS没有区别。区别在于背景像素填充部分。我认为Muller算法的核心（layer seperation）是确认背景部分是哪一部分，然后用统一的背景像素填充。这样做的好处是避免了inpainting，缺陷如IRCCyN/IVC DIBR image database所示，引入了拉伸（毛线，stretch）效果
	warping_1d(I_ref, D_ref, I_syn, D_syn);

	//test
	cv::imshow("before hole filling", I_syn);

	


	return true;
}

bool algorithm::warping_1d(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn)
{
	// 输入深度图必须是单通道的，因为我不会写通道检查

	std::vector<int> table(256);

	int knear = 0;
	int kfar = 128;
	int eye_seperation = 6;
	int view_distance = 200; //越小baseline越大，补洞的差异性越好，但是相应的，inpainting花销越大。=200的时候，telea算法
	int Npix = 320;
	int S = 25;
	cv::Size size = I_ref.size();
	//cout << size.height << " " << size.width << endl;
	float A, h;

	for (int i = 0; i < 256; i++)
	{
		A = i * (knear / 64 + kfar / 16) / 255.0;
		h = -eye_seperation * Npix * (A - kfar / 16) / view_distance;
		table[i] = static_cast<int>(h / 2);
	}

	//for (auto x : table)
	//{
	//	cout << x << " ";
	//}

	int depth_level;
	int shift;

	for (int i = 0; i < size.height; i++)
		for (int j = 0; j < size.width; j++)
		{
			depth_level = D_ref.at<uchar>(i, j);
			shift = table[depth_level];
			if (j + shift - S >= 0 && j + shift - S < size.width)
			{
				I_syn.at<cv::Vec3b>(i, j + shift - S) = I_ref.at<cv::Vec3b>(i, j);
				D_syn.at<uchar>(i, j + shift - S) = D_ref.at<uchar>(i, j);
			}
		}

	// test
	//cv::imshow("texture", I_syn);
	//cv::imshow("depth", D_syn);

	return true;
}

const cv::Mat algorithm::mask_detection(const cv::Mat & input)
{
	// 产生一张mask图，里面的像素都是黑白的
	cv::Mat mask = cv::Mat(input.size(), CV_8U);

	cv::Size size = input.size();

	for (int i = 0; i < size.height; i++)
		for (int j = 0; j < size.width; j++)
		{
			if (input.at<cv::Vec3b>(i, j)[0] == COLOR_BLACK && input.at<cv::Vec3b>(i, j)[1] == COLOR_BLACK && input.at<cv::Vec3b>(i, j)[2] == COLOR_BLACK)
				mask.at<uchar>(i, j) = COLOR_WHITE;
			else
				mask.at<uchar>(i, j) = COLOR_BLACK;
		}

	//test
	imshow("mask", mask);

	return mask;
}
