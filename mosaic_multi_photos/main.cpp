#include "mosaic.h"
using namespace std;
using namespace cv;
#define DisPlay 1

int main()
{
	mosaic mosaicGenerator;
	//遍历文件夹参数
	char fileName[50];
	string fileFolderPath = "..\\images";
	string fileExtension = "JPG";
	string fileFolder = fileFolderPath + "\\*." + fileExtension;
	struct _finddata_t fileInfo;    // 文件信息结构体
	//end
	int side_cols = 300;
	int side_rows = 300;
	double num1 = 0, num2 = 0, count = 0;
	Mat image1 = imread("..//first Image//DSC00237.JPG", 1);
	resize(image1, image1, Size(0, 0), 0.1, 0.1);
	Mat H;
	Mat result_tmp;
	image1.copyTo(result_tmp);
	//The first image is mosaic in the result
	cv::Mat result(1500, 1500, image1.type(), Scalar(0, 0, 0));
	cv::Mat half(result, cv::Rect(side_cols, side_rows, image1.cols, image1.rows));//确定第一张图的位置，这个必
	image1.copyTo(half);                                                          //须与计算H矩阵时使用的第一张图一致
	//End 
	long findResult = _findfirst(fileFolder.c_str(), &fileInfo);
	if (findResult == -1)
	{
		_findclose(findResult);
		return -1;
	}
	do
	{
		sprintf(fileName, "%s\\%s", fileFolderPath.c_str(), fileInfo.name);
		if (fileInfo.attrib == _A_ARCH)
		{
			Mat image_pce = imread(fileName, 1);
			resize(image_pce, image_pce, Size(0, 0), 0.1, 0.1);
			mosaicGenerator.findKeypoint(result_tmp, image_pce, H);
			mosaicGenerator.homography_warp(image_pce, H, result_tmp);
			//imshow(fileName, result_tmp);
#if DisPlay
			imshow("result_tmp", result_tmp);
			waitKey(0);
#endif
			for (vector<Mat_<double>>::const_iterator iter = mosaicGenerator.vec_translation.begin(); iter != mosaicGenerator.vec_translation.end(); ++iter)
			{
				Mat_<double> a = *iter;
				num1 += a.at<double>(0, 2);
				num2 += a.at<double>(1, 2);
				++count;
			}
			if (side_cols - num1 < 0 || side_rows - num2 < 0)
			{
				cout << "Error:side_rows and cols may be too small" << endl;
				return -2;
			}
			cout << "count is " << count << endl;
			cout << "\n\r" << endl;
			mosaicGenerator.addImage(result_tmp, result, side_cols - num1, side_rows - num2);
			num1 = 0;
			num2 = 0;
			count = 0;
		}
	} while (!_findnext(findResult, &fileInfo));
	imwrite("..//result.jpg", result);
#if DisPlay
	imshow("result image", result);
	waitKey(0);
#endif
	_findclose(findResult);
	return 0;
}
