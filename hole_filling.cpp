#include "hole_filling.h"
#include <time.h>
#include <vector>

#define COLOR_WHITE 255
#define COLOR_BLACK 0

// utility functions needed for inpainting



/*
* Return a % b where % is the mathematical modulus operator.
*/
int algorithm::mod(int a, int b) {
	return ((a % b) + b) % b;
}


/*
* Load the color, mask, grayscale images with a border of size
* radius around every image to prevent boundary collisions when taking patches
*/
void algorithm::loadInpaintingImages(
	cv::Mat& colorMat,
	cv::Mat& maskMat,
	cv::Mat& grayMat)
{
	// convert colorMat to depth CV_32F for colorspace conversions
	colorMat.convertTo(colorMat, CV_32F);
	colorMat /= 255.0f;

	// add border around colorMat
	cv::copyMakeBorder(
		colorMat,
		colorMat,
		RADIUS,
		RADIUS,
		RADIUS,
		RADIUS,
		cv::BORDER_CONSTANT,
		cv::Scalar_<float>(0, 0, 0)
	);

	cv::cvtColor(colorMat, grayMat, CV_BGR2GRAY);
}


/*
* Show a Mat object quickly. For testing purposes only.
*/

void algorithm::showMat(const cv::String& winname, const cv::Mat& mat, int time/*= 5*/)
{
	cv::namedWindow(winname);
	cv::imshow(winname, mat);
	cv::imwrite("virtual_criminisi.png", mat);
	cv::waitKey(time);
	cv::destroyWindow(winname);
}


/*
* Extract closed boundary from mask.
*/
void algorithm::getContours(const cv::Mat& mask,
	contours_t& contours,
	hierarchy_t& hierarchy
)
{
	cv::findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
}


/*
* Get a patch of size RAIDUS around point p in mat.
*/
cv::Mat algorithm::getPatch(const cv::Mat& mat, const cv::Point& p)
{
	return  mat(
		cv::Range(p.y - RADIUS, p.y + RADIUS + 1),
		cv::Range(p.x - RADIUS, p.x + RADIUS + 1)
	);
}


// get the x and y derivatives of a patch centered at patchCenter in image
// computed using a 3x3 Scharr filter
void algorithm::getDerivatives(const cv::Mat& grayMat, cv::Mat& dx, cv::Mat& dy)
{
	cv::Sobel(grayMat, dx, -1, 1, 0, -1);
	cv::Sobel(grayMat, dy, -1, 0, 1, -1);
}


/*
* Get the unit normal of a dense list of boundary points centered around point p.
*/
cv::Point2f algorithm::getNormal(const contour_t& contour, const cv::Point& point)
{
	int sz = (int)contour.size();

	int pointIndex = (int)(std::find(contour.begin(), contour.end(), point) - contour.begin());

	if (sz == 1)
	{
		return cv::Point2f(1.0f, 0.0f);
	}
	else if (sz < 2 * BORDER_RADIUS + 1)
	{
		// Too few points in contour to use LSTSQ regression
		// return the normal with respect to adjacent neigbourhood
		cv::Point adj = contour[(pointIndex + 1) % sz] - contour[pointIndex];
		return cv::Point2f(adj.y, -adj.x) / cv::norm(adj);
	}

	// Use least square regression
	// create X and Y mat to SVD
	cv::Mat X(cv::Size(2, 2 * BORDER_RADIUS + 1), CV_32F);
	cv::Mat Y(cv::Size(1, 2 * BORDER_RADIUS + 1), CV_32F);

	int i = mod((pointIndex - BORDER_RADIUS), sz);

	float* Xrow;
	float* Yrow;

	int count = 0;
	int countXequal = 0;
	while (count < 2 * BORDER_RADIUS + 1)
	{
		Xrow = X.ptr<float>(count);
		Xrow[0] = contour[i].x;
		Xrow[1] = 1.0f;

		Yrow = Y.ptr<float>(count);
		Yrow[0] = contour[i].y;

		if (Xrow[0] == contour[pointIndex].x)
		{
			++countXequal;
		}

		i = mod(i + 1, sz);
		++count;
	}

	if (countXequal == count)
	{
		return cv::Point2f(1.0f, 0.0f);
	}
	// to find the line of best fit
	cv::Mat sol;
	cv::solve(X, Y, sol, cv::DECOMP_SVD);

	float slope = sol.ptr<float>(0)[0];
	cv::Point2f normal(-slope, 1);

	return normal / cv::norm(normal);
}


/*
* Return the confidence of confidencePatch
*/
double algorithm::computeConfidence(const cv::Mat& confidencePatch)
{
	return cv::sum(confidencePatch)[0] / (double)confidencePatch.total();
}


/*
* Iterate over every contour point in contours and compute the
* priority of path centered at point using grayMat and confidenceMat
*/
void algorithm::computePriority(const contours_t& contours, const cv::Mat& grayMat, const cv::Mat& confidenceMat, cv::Mat& priorityMat)
{

	// define some patches
	cv::Mat confidencePatch;
	cv::Mat magnitudePatch;

	cv::Point2f normal;
	cv::Point maxPoint;
	cv::Point2f gradient;

	double confidence;

	// get the derivatives and magnitude of the greyscale image
	cv::Mat dx, dy, magnitude;
	getDerivatives(grayMat, dx, dy);
	cv::magnitude(dx, dy, magnitude);

	// mask the magnitude
	cv::Mat maskedMagnitude(magnitude.size(), magnitude.type(), cv::Scalar_<float>(0));
	magnitude.copyTo(maskedMagnitude, (confidenceMat != 0.0f));
	cv::erode(maskedMagnitude, maskedMagnitude, cv::Mat());

	// for each point in contour
	cv::Point point;

	for (int i = 0; i < contours.size(); ++i)
	{
		contour_t contour = contours[i];

		for (int j = 0; j < contour.size(); ++j)
		{

			point = contour[j];

			confidencePatch = getPatch(confidenceMat, point);

			// get confidence of patch
			confidence = cv::sum(confidencePatch)[0] / (double)confidencePatch.total();
			assert(0 <= confidence && confidence <= 1.0f);

			// get the normal to the border around point
			normal = getNormal(contour, point);

			// get the maximum gradient in source around patch
			magnitudePatch = getPatch(maskedMagnitude, point);
			cv::minMaxLoc(magnitudePatch, NULL, NULL, NULL, &maxPoint);
			gradient = cv::Point2f(
				-getPatch(dy, point).ptr<float>(maxPoint.y)[maxPoint.x],
				getPatch(dx, point).ptr<float>(maxPoint.y)[maxPoint.x]
			);

			// set the priority in priorityMat
			priorityMat.ptr<float>(point.y)[point.x] = std::abs((float)confidence * gradient.dot(normal));
			assert(priorityMat.ptr<float>(point.y)[point.x] >= 0);
		}
	}
}


/*
* Transfer the values from patch centered at psiHatQ to patch centered at psiHatP in
* mat according to maskMat.
*/
void algorithm::transferPatch(const cv::Point& psiHatQ, const cv::Point& psiHatP, cv::Mat& mat, const cv::Mat& maskMat)
{
	// copy contents of psiHatQ to psiHatP with mask
	getPatch(mat, psiHatQ).copyTo(getPatch(mat, psiHatP), getPatch(maskMat, psiHatP));
}

/*
* Runs template matching with tmplate and mask tmplateMask on source.
* Resulting Mat is stored in result.
*
*/
cv::Mat algorithm::computeSSD(const cv::Mat& tmplate, const cv::Mat& source, const cv::Mat& tmplateMask)
{
	cv::Mat result(source.rows - tmplate.rows + 1, source.cols - tmplate.cols + 1, CV_32F, 0.0f);

	cv::matchTemplate(source,
		tmplate,
		result,
		CV_TM_SQDIFF,
		tmplateMask
	);
	cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
	cv::copyMakeBorder(result, result, RADIUS, RADIUS, RADIUS, RADIUS, cv::BORDER_CONSTANT, 1.1f);

	return result;
}

bool algorithm::Fehn_interpolation(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn)
{
	// 这里改变了传入参考视点深度图像的类型（前提是深度图是单通道的）

	cv::GaussianBlur(D_ref, D_ref, cv::Size(3, 3), 0, 0);

	warping_1d(I_ref, D_ref, I_syn, D_syn);

	// 裁剪掉边框
	int s = 25; // 默认裁剪掉的边的宽度
	cv::Rect myROI(s, 0, I_syn.size().width - 2 * s, I_syn.size().height);

	cv::Mat texture_mask, depth_mask;

	// 颜色图和深度图都裁剪掉边框
	I_syn(myROI).copyTo(texture_mask);
	D_syn(myROI).copyTo(depth_mask);

	cv::resize(texture_mask, texture_mask, I_syn.size(), 0, 0, cv::INTER_NEAREST);
	cv::resize(depth_mask, depth_mask, D_syn.size(), 0, 0, cv::INTER_NEAREST);

	cv::medianBlur(texture_mask, texture_mask, 3);
	//cv::imshow("mask", mask);

	texture_mask.copyTo(I_syn);
	depth_mask.copyTo(D_syn);

	cv::Size size = I_syn.size();

	if (getDistance() > 0)
		for (int i = 0; i < size.height; i++)
			for (int j = 0; j < size.width; j++)
			{
				if (I_syn.at<cv::Vec3b>(i, j)[0] == COLOR_BLACK && I_syn.at<cv::Vec3b>(i, j)[1] == COLOR_BLACK && I_syn.at<cv::Vec3b>(i, j)[2] == COLOR_BLACK)
				{
					I_syn.at<cv::Vec3b>(i, j) = I_syn.at<cv::Vec3b>(i, j - 1);
				}
			}
	else
		for (int i = 0; i < size.height; i++)
			for (int j = size.width - 1; j >= 0; j--)
			{
				if (I_syn.at<cv::Vec3b>(i, j)[0] == COLOR_BLACK && I_syn.at<cv::Vec3b>(i, j)[1] == COLOR_BLACK && I_syn.at<cv::Vec3b>(i, j)[2] == COLOR_BLACK)
				{
					I_syn.at<cv::Vec3b>(i, j) = I_syn.at<cv::Vec3b>(i, j + 1);
				}
			}

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
	bool flag = warping_1d(I_ref, D_ref, I_syn, D_syn);

	//test
	cv::imshow("before hole filling", I_syn);

	cv::Size size = I_syn.size();

	if (flag) //flag==true, 空洞出现在右边
	{
		for (int i = 0; i < size.height; i++)
				for (int j = 0; j < size.width; j++)
				{
					if (I_syn.at<cv::Vec3b>(i, j)[0] == COLOR_BLACK && I_syn.at<cv::Vec3b>(i, j)[1] == COLOR_BLACK && I_syn.at<cv::Vec3b>(i, j)[1] == COLOR_BLACK)
						I_syn.at<cv::Vec3b>(i, j) = I_syn.at<cv::Vec3b>(i, j - 1);
				}
	}
	else
	{
		for (int i = 0; i < size.height; i++)
			for (int j = size.width; j > 0; j--)
			{
				if (I_syn.at<cv::Vec3b>(i, j)[0] == COLOR_BLACK && I_syn.at<cv::Vec3b>(i, j)[1] == COLOR_BLACK && I_syn.at<cv::Vec3b>(i, j)[1] == COLOR_BLACK)
					I_syn.at<cv::Vec3b>(i, j) = I_syn.at<cv::Vec3b>(i, j + 1);
			}
	}

	//test
	//cv::imshow("after holefilling", I_syn);

	return true;
}

bool algorithm::Ndijiki(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn)
{
	warping_1d(I_ref, D_ref, I_syn, D_syn);

	
	// ---------------read the images-------------------
	// colorMat - color picture + border
	// maskMat - mask picture + border
	// grayMat - gray picture + border
	cv::Mat colorMat;
	I_syn.copyTo(colorMat);
	cv::Mat grayMat;
	cv::Mat maskMat  = mask_detection(colorMat);
	cv::bitwise_not(maskMat, maskMat);
	//test
	//cv::imshow("maskMat", maskMat);
	loadInpaintingImages(colorMat, maskMat, grayMat);

	// confidenceMat - confidence picture + border
	cv::Mat confidenceMat;
	maskMat.convertTo(confidenceMat, CV_32F);
	confidenceMat /= 255.0;

	// add borders around maskMat and confidenceMat
	cv::copyMakeBorder(maskMat, maskMat,
		RADIUS, RADIUS, RADIUS, RADIUS,
		cv::BORDER_CONSTANT, 255);
	cv::copyMakeBorder(confidenceMat, confidenceMat,
		RADIUS, RADIUS, RADIUS, RADIUS,
		cv::BORDER_CONSTANT, 0.0001f);

	// -------------------start Criminisi inpainting----------------------
	contours_t contours; //mask contours
	hierarchy_t hierarchy; // contours hierarchy

	// priorityMat - priority values for all contour points + border
	cv::Mat priorityMat(confidenceMat.size(), CV_32FC1); // priority value matrix for each contour point

	cv::Point psiHatP; //psiHatP - point of highest confidence

	cv::Mat psiHatPColor; // color patch around psiHatP

	cv::Mat psiHatPConfidence; // confidence patch around psiHatP
	double confidence; // confidence of psiHatPConfidence

	cv::Point psiHatQ; // psiHatQ - point of closest patch

	cv::Mat result; // holds result from template matching
	cv::Mat erodedMask; // eroded mask

	cv::Mat templateMask; // mask for template match (3 channel)

	// eroded mask is used to ensure that psiHatQ is not overlapping with target
	cv::erode(maskMat, erodedMask, cv::Mat(), cv::Point(-1, -1), RADIUS);

	// main loop
	const size_t area = maskMat.total();

	while (cv::countNonZero(maskMat) != area) // end when target is filled
	{
		// set priority matrix to -.1, lower that 0 so that border area is nevel selected
		priorityMat.setTo(-0.1f);

		// get the contours of mask
		getContours((maskMat == 0), contours, hierarchy);

		// compute the priority for all contour points
		computePriority(contours, grayMat, confidenceMat, priorityMat);

		// get the patch with the greatest priority
		cv::minMaxLoc(priorityMat, NULL, NULL, NULL, &psiHatP);
		psiHatPColor = getPatch(colorMat, psiHatP);
		psiHatPConfidence = getPatch(confidenceMat, psiHatP);

		cv::Mat confInv = (psiHatPConfidence != 0.0f);
		confInv.convertTo(confInv, CV_32F);
		confInv /= 255.0f;
		// get the patch in source with least distance to psiHatPColor wrt source of psiHatP
		cv::Mat mergeArrays[3] = { confInv, confInv, confInv };
		cv::merge(mergeArrays, 3, templateMask);
		result = computeSSD(psiHatPColor, colorMat, templateMask);

		// set all target regions to 1.1, which is over the maximum value possible from SSD
		result.setTo(1.1f, erodedMask == 0);
		// get the minimum point of SSD between psiHatPColor and colorMat
		cv::minMaxLoc(result, NULL, NULL, &psiHatQ);

		// updates
		// copy from psiHatQ to psiHatP for each colorspace
		transferPatch(psiHatQ, psiHatP, grayMat, (maskMat == 0));
		transferPatch(psiHatQ, psiHatP, colorMat, (maskMat == 0));

		// fill in confidenceMat with confidences C(pixel) = C(psiHatP)
		confidence = computeConfidence(psiHatPConfidence);
		// update confidence
		psiHatPConfidence.setTo(confidence, (psiHatPConfidence == 0.0f));
		// update maskMat
		maskMat = (confidenceMat != 0.0f);
	}

	cv::Rect border = cv::Rect(RADIUS, RADIUS, colorMat.size().width - 2 * RADIUS, colorMat.size().height - 2 * RADIUS);
	I_syn = colorMat(border);

	// test
	//cv::imshow("final result", I_syn);
	//cv::imshow("criminit", I_syn);
	//std::cout << colorMat.size();
	return true;
}

bool algorithm::warping_1d(const cv::Mat & I_ref, const cv::Mat & D_ref, cv::Mat & I_syn, cv::Mat & D_syn)
{
	// 输入深度图必须是单通道的，因为我不会写通道检查

	std::vector<int> table(256);

	double knear, kfar;
	cv::Point pnear, pfar;
	cv::minMaxLoc(D_ref, &kfar, &knear, &pfar, &pnear);

	//knear = 255; kfar = 0;

	//test
	//std::cout << knear << " " << kfar << std::endl;

	int view_distance = getDistance(); //越小baseline越大，补洞的差异性越好，但是相应的，inpainting花销越大。=200的时候，telea算法
	int S = 0;
	cv::Size size = I_ref.size();
	//cout << size.height << " " << size.width << endl;
	float A, h;

	for (int i = 0; i < 256; i++)
	{
		A = static_cast<float>(i - kfar) / (knear - kfar);
		h = A * view_distance;
		table[i] = static_cast<int>(h);
	}

	//for (auto x : table)
	//{
	//	std::cout << x << " ";
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
				if (I_syn.at<cv::Vec3b>(i, j + shift - S)[0] == 0 && I_syn.at<cv::Vec3b>(i, j + shift - S)[1] == 0 && I_syn.at<cv::Vec3b>(i, j + shift - S)[2] == 0)
				{
					I_syn.at<cv::Vec3b>(i, j + shift - S) = I_ref.at<cv::Vec3b>(i, j);
					D_syn.at<uchar>(i, j + shift - S) = D_ref.at<uchar>(i, j);
				}
				else if (D_syn.at<uchar>(i, j + shift - S) < D_ref.at<uchar>(i, j))
				{
					I_syn.at<cv::Vec3b>(i, j + shift - S) = I_ref.at<cv::Vec3b>(i, j);
					D_syn.at<uchar>(i, j + shift - S) = D_ref.at<uchar>(i, j);
				}
			}
		}

	// test
	//cv::imshow("after 3D warping", I_syn);
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
	//imshow("mask", mask);

	return mask;
}

