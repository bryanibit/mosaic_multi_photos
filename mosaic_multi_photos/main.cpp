#include <iostream>
#include<vector>
#include <math.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
using namespace std;
using namespace cv;
Mat_<double> translation;
vector<Mat_<double>> vec_translation;
void to_homogeneous(const std::vector< cv::Point2f >& non_homogeneous, std::vector< cv::Point3f >& homogeneous)
{
	homogeneous.resize(non_homogeneous.size());
	for (size_t i = 0; i < non_homogeneous.size(); i++) {
		homogeneous[i].x = non_homogeneous[i].x;
		homogeneous[i].y = non_homogeneous[i].y;
		homogeneous[i].z = 1.0;
	}
}

// Convert a vector of homogeneous 2D points to a vector of non-homogenehous 2D points.
void from_homogeneous(const std::vector< cv::Point3f >& homogeneous, std::vector< cv::Point2f >& non_homogeneous)
{
	non_homogeneous.resize(homogeneous.size());
	for (size_t i = 0; i < non_homogeneous.size(); i++) {
		non_homogeneous[i].x = homogeneous[i].x / homogeneous[i].z;
		non_homogeneous[i].y = homogeneous[i].y / homogeneous[i].z;
	}
}

// Transform a vector of 2D non-homogeneous points via an homography.
std::vector<cv::Point2f> transform_via_homography(const std::vector<cv::Point2f>& points, const cv::Matx33f& homography)
{
	std::vector<cv::Point3f> ph;
	to_homogeneous(points, ph);
	for (size_t i = 0; i < ph.size(); i++) {
		ph[i] = homography*ph[i];
	}
	std::vector<cv::Point2f> r;
	from_homogeneous(ph, r);
	return r;
}

// Find the bounding box of a vector of 2D non-homogeneous points.
cv::Rect_<float> bounding_box(const std::vector<cv::Point2f>& p)
{
	cv::Rect_<float> r;
	float x_min = std::min_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.x < rhs.x; })->x;
	float x_max = std::max_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.x < rhs.x; })->x;
	float y_min = std::min_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.y < rhs.y; })->y;
	float y_max = std::max_element(p.begin(), p.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {return lhs.y < rhs.y; })->y;
	return cv::Rect_<float>(x_min, y_min, x_max - x_min, y_max - y_min);
}

// Warp the image src into the image dst through the homography H.
// The resulting dst image contains the entire warped image, this
// behaviour is the same of Octave's imperspectivewarp (in the 'image'
// package) behaviour when the argument bbox is equal to 'loose'.
// See http://octave.sourceforge.net/image/function/imperspectivewarp.html
void homography_warp(const cv::Mat& src, const cv::Mat& H, cv::Mat& dst)
{
	std::vector< cv::Point2f > corners;
	corners.push_back(cv::Point2f(0, 0));
	corners.push_back(cv::Point2f(src.cols, 0));
	corners.push_back(cv::Point2f(0, src.rows));
	corners.push_back(cv::Point2f(src.cols, src.rows));

	std::vector< cv::Point2f > projected = transform_via_homography(corners, H);
	cv::Rect_<float> bb = bounding_box(projected);

	translation = (cv::Mat_<double>(3, 3) << 1, 0, -bb.tl().x, 0, 1, -bb.tl().y, 0, 0, 1);
	vec_translation.push_back(translation);
	cv::warpPerspective(src, dst, translation*H, bb.size());
}
void findKeypoint(Mat image_main, Mat image_, Mat & H)
{
	//image_main为主图像，此函数返回的H为image转换为某矩阵可以和主矩阵相加而重合
	Mat gray_image1, gray_image2;
	if (!image_main.data || !image_.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return;
	}

	// Convert to Grayscale
	cvtColor(image_main, gray_image1, CV_RGB2GRAY);
	cvtColor(image_, gray_image2, CV_RGB2GRAY);

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	SurfFeatureDetector detector(minHessian);

	std::vector< KeyPoint > keypoints_object, keypoints_scene;

	detector.detect(gray_image2, keypoints_object);
	detector.detect(gray_image1, keypoints_scene);

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute(gray_image2, keypoints_object, descriptors_object);
	extractor.compute(gray_image1, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	std::vector< Point2f > obj;
	std::vector< Point2f > scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	// Find the Homography Matrix
	H = findHomography(obj, scene, CV_RANSAC);
	// Use the Homography Matrix to warp the images
	//cout << H << endl;
}
void addImage(Mat & image, Mat & container, int side_cols, int side_rows)
{
	//image为经过透视变换后的矩阵，加到container上，结果还放在container中
	Mat container_roi = container(Rect(side_cols, side_rows, image.cols, image.rows));
	Mat greyImage, greyContainer;
	Mat thresholdImage, thresholdContainer, intersect, mask, maskImage;
	cvtColor(image, greyImage, CV_RGB2GRAY);
	/*imshow("image", greyImage);
	imwrite("greyImage.jpg", greyImage);
	waitKey(0);*/
	cvtColor(container_roi, greyContainer, CV_RGB2GRAY);
	threshold(greyImage, thresholdImage, 10, 255, THRESH_BINARY);
	threshold(greyContainer, thresholdContainer, 10, 255, THRESH_BINARY);
	bitwise_and(thresholdImage, thresholdContainer, intersect);
	subtract(thresholdImage, intersect, mask);
	Mat kernel(3, 3, CV_8UC1);
	dilate(mask, mask, kernel);
	/*imshow("kernel", kernel);
	waitKey(0);*/
	bitwise_and(image, image, maskImage, mask);
	add(container_roi, maskImage, container_roi);
	/*imshow("container", container);
	imwrite("container_roi.jpg", container_roi);
	waitKey(0);*/
}
int main()
{
	int side_cols = 300;
	int side_rows = 300;
	double num1 = 0, num2 = 0, count = 0;
	Mat image1 = imread("images//DSC00237.JPG", 1);
	Mat image2 = imread("images//DSC00247.JPG", 1);
	Mat image3 = imread("images//DSC00275.JPG", 1);
	Mat image4 = imread("images//DSC00276.JPG", 1);
	resize(image1, image1, Size(0, 0), 0.1, 0.1);
	resize(image2, image2, Size(0, 0), 0.1, 0.1);
	resize(image3, image3, Size(0, 0), 0.1, 0.1);
	resize(image4, image4, Size(0, 0), 0.1, 0.1);
	Mat H, M;
	Mat result_tmp, result3, result4;
	image1.copyTo(result_tmp);
	imshow("result_tmp", result_tmp);
	waitKey(0);
	//The first image is mosaic in the result
	cv::Mat result(1500, 1500, image1.type(), Scalar(0, 0, 0));
	cv::Mat half(result, cv::Rect(side_cols, side_rows, image1.cols, image1.rows));//确定第一张图的位置，这个必
	image1.copyTo(half);                                                          //须与计算H矩阵时使用的第一张图一致
	imshow("result image1", result);
	waitKey(2);
	//End 
	for (int i = 1; i <= 1; i++)
	{
		findKeypoint(result_tmp, image2, H);
		homography_warp(image2, H, result_tmp);
		for (vector<Mat_<double>>::const_iterator iter = vec_translation.begin(); iter != vec_translation.end(); ++iter)
		{
			Mat_<double> a = *iter;
			num1 += a.at<double>(0, 2);
			num2 += a.at<double>(1, 2);
			++count;
		}
		if (side_cols - num1 < 0 || side_rows - num2 < 0)
		{
			cout << "Error:side_rows and cols may be too small" << endl;
			return -1;
		}
		cout << "count is " << count << endl;
		addImage(result_tmp, result, side_cols - num1, side_rows - num2);
		imshow("test result is ", result);
		waitKey(0);
		num1 = 0;
		num2 = 0;
		count = 0;
	}
	return 0;
}