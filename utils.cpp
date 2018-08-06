#include "utils.h"

void element::readVideo(std::string texture_filename, std::string depth_filename)
{
	cv::VideoCapture texture_capture(texture_filename);
	cv::VideoCapture depth_capture(depth_filename);

	frame = 0;

	while (true)
	{
		cv::Mat texture_image, depth_image;
		texture_capture >> texture_image;
		depth_capture >> depth_image;

		if (texture_image.empty() || depth_image.empty())
			break;

		cv::cvtColor(depth_image, depth_image, CV_RGB2GRAY);

		texture_video.push_back(texture_image);
		depth_video.push_back(depth_image);

		frame++;
	}
	//test frame here!!!
	//frame = 2;

	std::cout << "Total frames: " << frame << std::endl;

	// test
	//showVideo();

	texture_capture.release();
	depth_capture.release();
}

void element::writeVideo(std::string texture_filename, std::string depth_filename)
{
	int fourcc = CV_FOURCC('I', 'Y', 'U', 'V');

	cv::VideoWriter texture_writer(texture_filename, fourcc, 25.0, cv::Size(cols, rows));
	cv::VideoWriter depth_writer(depth_filename, fourcc, 25.0, cv::Size(cols, rows), false);

	for (int i = 0; i < frame; i++)
	{
		cv::Mat texture_image = texture_video.at(i);
		cv::Mat depth_image = depth_video.at(i);
		texture_writer << texture_image;
		depth_writer << depth_image;
		//if (i % 50 == 0)
		//{
		//	std::cout << "Write " << i << " frame" << std::endl;
		//}	
	}

	texture_writer.release();
	depth_writer.release();
}

void element::writeVideo(std::string texture_filename) {
	int fourcc = CV_FOURCC('I', 'Y', 'U', 'V');
	cv::VideoWriter texture_writer(texture_filename, fourcc, 25.0, cv::Size(cols, rows));
	for (int i = 0; i < frame; i++)
	{
		cv::Mat texture_image = texture_video.at(i);
		cv::Mat depth_image = depth_video.at(i);
		texture_writer << texture_image;
		//if (i % 50 == 0)
		//{
		//	std::cout << "Write " << i << " frame" << std::endl;
		//}
	}

	texture_writer.release();
}

void element::processVideo(const int distance, const int type)
{
	algorithm al(distance);
	showAlgorithmType(type);

	for (int i = 0; i < frame; i++)
	{
		cv::Mat I_ref = texture_video.at(i);
		cv::Mat D_ref = depth_video.at(i);
		cv::Mat I_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Mat D_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8U, cv::Scalar(0));


		//clock_t start = clock();
		switch (type)
		{
		case A1:
			al.Fehn_interpolation(I_ref, D_ref, I_syn, D_syn);
			break;
		case A2:
			al.Fehn_inpainting(I_ref, D_ref, I_syn, D_syn);
			break;
		case A3:
			al.Tanimoto(I_ref, D_ref, I_syn, D_syn);
			break;
		case A4:
			al.Muller(I_ref, D_ref, I_syn, D_syn);
			break;
		case A5:
			al.Ndijiki(I_ref, D_ref, I_syn, D_syn);
			break;
		case A7:
			al.warping_1d(I_ref, D_ref, I_syn, D_syn);
			break;
		default:
			break;
		}

		
		//clock_t ends = clock();

		//std::cout << "Running time of 3D warping (ms): " << static_cast<double>(ends - start) / CLOCKS_PER_SEC * 1000 << std::endl;

		texture_video.at(i) = I_syn;
		depth_video.at(i) = D_syn;

		//if (i % 50 == 0)
		//{
		//	std::cout << "Processed " << i << " frames." << std::endl;
		//}
	}
}

void element::showVideo()
{
	for (auto texture_image : texture_video)
	{
		cv::imshow("Video", texture_image);
		cv::waitKey(30);
	}

	for (auto depth_image : depth_video)
	{
		cv::imshow("Video", depth_image);
		cv::waitKey(30);
	}
}

void element::showAlgorithmType(int type)
{
	switch (type)
	{
	case A1:
		std::cout << "DIBR Algorithm: A1-Fehn_interpolation" << std::endl;
		break;
	case A2:
		std::cout << "DIBR Algorithm: A2-Fehn_inpainting" << std::endl;
		break;
	case A3:
		std::cout << "DIBR Algorithm: A3-Tamino" << std::endl;
		break;
	case A4:
		std::cout << "DIBR Algorithm: A4-Muller" << std::endl;
		break;
	case A5:
		std::cout << "DIBR Algorithm: A5-Ndijsk" << std::endl;
		break;
	case A6:
		std::cout << "DIBR Algorithm: A6-Kopper" << std::endl;
		break;
	case A7:
		std::cout << "DIBR Algorithm: A7-No_holefilling" << std::endl;
		break;
	default:
		break;
	}
}

void element::processVideo(int distance, std::string t_file, std::string d_file)
{
	//process the specific video
	kooper k(t_file, d_file, distance);
	std::cout << "DIBR Algorithm: A6-Kopper" << std::endl;

	for (int i = 0; i < frame; i++) {
		cv::Mat I_ref = texture_video.at(i);
		cv::Mat D_ref = depth_video.at(i);
		cv::Mat I_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Mat D_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8U, cv::Scalar(0));

		/*clock_t start = clock();*/

		I_syn = k.A6Porcess(I_ref, D_ref);

		/*clock_t ends = clock();*/

		texture_video.at(i) = I_syn;

		/*std::cout << "Running frames: " << i << " of kooper porcess (ms): " << static_cast<double>(ends - start) / CLOCKS_PER_SEC * 1000 << std::endl;*/

	}

}

