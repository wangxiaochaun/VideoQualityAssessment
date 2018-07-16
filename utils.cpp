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
	//test for algorithm
	//frame = 10;

	std::cout << "Total frames: " << frame << std::endl;

	// test
	//showVideo();

	texture_capture.release();
	depth_capture.release();
}

void element::writeVideo(std::string texture_filename, std::string depth_filename)
{
	cv::VideoWriter texture_writer(texture_filename, -1, 25.0, cv::Size(cols, rows));
	cv::VideoWriter depth_writer(depth_filename, -1, 25.0, cv::Size(cols, rows), false);

	for (int i = 0; i < frame; i++)
	{
		cv::Mat texture_image = texture_video.at(i);
		cv::Mat depth_image = depth_video.at(i);
		texture_writer << texture_image;
		depth_writer << depth_image;
		std::cout << "Write " << i << " frame" << std::endl;
	}

	texture_writer.release();
	depth_writer.release();
}

void element::writeVideo(std::string texture_filename) 
{
	cv::VideoWriter texture_writer(texture_filename, -1, 25.0, cv::Size(cols, rows));
	for (int i = 0; i < frame; i++)
	{
		cv::Mat texture_image = texture_video.at(i);
		texture_writer << texture_image;
		std::cout << "Write " << i << " frame" << std::endl;
	}
	texture_writer.release();
}

void element::processVideo(const int distance, const int type)
{
	algorithm al(distance);
	for (int i = 0; i < frame; i++)
	{
		cv::Mat I_ref = texture_video.at(i);
		cv::Mat D_ref = depth_video.at(i);
		cv::Mat I_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Mat D_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8U, cv::Scalar(0));

		clock_t start = clock();
		switch (type)
		{
		case A1:
			al.Fehn_interpolation(I_ref, D_ref, I_syn, D_syn);
		default:
			break;
		}
		
		clock_t ends = clock();

		std::cout << "Running time of 3D warping (ms): " << static_cast<double>(ends - start) / CLOCKS_PER_SEC * 1000 << std::endl;

		texture_video.at(i) = I_syn;
		depth_video.at(i) = D_syn;
	}
}

void element::processVideo(int distance, std::string t_file, std::string d_file)
{
	//process the specific video
	kooper k(t_file, d_file,distance);

	for (int i = 0; i < frame; i++) {
		cv::Mat I_ref = texture_video.at(i);
		cv::Mat D_ref = depth_video.at(i);
		cv::Mat I_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Mat D_syn = cv::Mat(I_ref.size().height, I_ref.size().width, CV_8U, cv::Scalar(0));

		clock_t start = clock();

		I_syn = k.A6Porcess(I_ref, D_ref);

		clock_t ends = clock();

		texture_video.at(i) = I_syn;

		std::cout << "Running frames: "<<i<<" of kooper porcess (ms): " << static_cast<double>(ends - start) / CLOCKS_PER_SEC * 1000 << std::endl;

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


