#include "hole_filling.h"
#include <time.h>
#include <vector>

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
//	cv::imshow("mask", mask);

	mask.copyTo(I_syn);

	return true;
}

bool algorithm::warping_1d(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn)
{
	// �������ͼ�����ǵ�ͨ���ģ���Ϊ�Ҳ���дͨ�����

	std::vector<int> table(256);

	int knear = 0;
	int kfar = 128;
	int eye_seperation = 6;
	int view_distance = 800;
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
