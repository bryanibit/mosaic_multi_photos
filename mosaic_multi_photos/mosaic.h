#include<vector>
#include <math.h>
#include <iostream>
#include <io.h>
#include <string>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
using namespace cv;
using namespace std;
class mosaic
{
public:
	Mat_<double> translation;
	vector<Mat_<double>> vec_translation;
	mosaic();
	~mosaic();
	void homography_warp(const cv::Mat& src, const cv::Mat& H, cv::Mat& dst);
	void findKeypoint(Mat image_main, Mat image_, Mat & H);
	void addImage(Mat & image, Mat & container, int side_cols, int side_rows);
private:
	
	// ×Óº¯Êý
	void to_homogeneous(const std::vector< cv::Point2f >& non_homogeneous, std::vector< cv::Point3f >& homogeneous);
	void from_homogeneous(const std::vector< cv::Point3f >& homogeneous, std::vector< cv::Point2f >& non_homogeneous);
	std::vector<cv::Point2f> transform_via_homography(const std::vector<cv::Point2f>& points, const cv::Matx33f& homography);
	cv::Rect_<float> bounding_box(const std::vector<cv::Point2f>& p);
	
	//end
};