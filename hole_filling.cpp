#include "hole_filling.h"
#include <time.h>
#include <vector>

#define COLOR_WHITE 255
#define COLOR_BLACK 0

bool algorithm::Fehn_interpolation(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn)
{
	// ����ı��˴���ο��ӵ����ͼ������ͣ�ǰ�������ͼ�ǵ�ͨ���ģ�

	cv::GaussianBlur(D_ref, D_ref, cv::Size(3, 3), 0, 0);

	warping_1d(I_ref, D_ref, I_syn, D_syn);

	// �ü����߿�
	int s = 25; // Ĭ�ϲü����ıߵĿ��
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
	// �������൱��ʱ�����800ms���ң�

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
	// VSRS�㷨�ļ򻯰汾��Ŀǰ����ô���ģ����ȣ���warping������ͼ������˫���˲�����Ȼ��������ֵѡ�񱳾������޲��ն����������inpainting��һ�飨inpainting̫����ʱ�����Կ�����Ƿ���ã�
	warping_1d(I_ref, D_ref, I_syn, D_syn);

	//test
	cv::imshow("before background filling", I_syn);
	//cv::imshow("D_syn", D_syn);

	cv::Mat temp = cv::Mat(D_syn.size(), CV_8U);

	// �������������˲�����С��������˲���ִ��Ч�ʵͣ�����ȡ9*9�� d=0ʱ���˲�����С�ɺ���������˹��sigma���������ġ�������ֵ��Ϳ����˲��˵�sigma��һ��ȡ����10
	cv::bilateralFilter(D_syn, temp, 9, 25, 25);

	//test
	//cv::imshow("After bilateral filter", temp);

	cv::Size size = I_syn.size();
	int threshold = 10; //��Ȳ���ֵ���ɵ�

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
	// Muller������VSRSһ�������������ӵ㡣��������ֻ�����򻯡�������ֺ�VSRSû�������������ڱ���������䲿�֡�����ΪMuller�㷨�ĺ��ģ�layer seperation����ȷ�ϱ�����������һ���֣�Ȼ����ͳһ�ı���������䡣�������ĺô��Ǳ�����inpainting��ȱ����IRCCyN/IVC DIBR image database��ʾ�����������죨ë�ߣ�stretch��Ч��
	warping_1d(I_ref, D_ref, I_syn, D_syn);

	//test
	cv::imshow("before hole filling", I_syn);

	


	return true;
}

bool algorithm::warping_1d(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn)
{
	// �������ͼ�����ǵ�ͨ���ģ���Ϊ�Ҳ���дͨ�����

	std::vector<int> table(256);

	int knear = 0;
	int kfar = 128;
	int eye_seperation = 6;
	int view_distance = 200; //ԽСbaselineԽ�󣬲����Ĳ�����Խ�ã�������Ӧ�ģ�inpainting����Խ��=200��ʱ��telea�㷨
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
	// ����һ��maskͼ����������ض��Ǻڰ׵�
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
