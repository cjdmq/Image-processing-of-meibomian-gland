#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<string>
#include<vector>
#include<time.h>
#include<algorithm>
#include<stdio.h>
#include<opencv2/core.hpp>
#include"B-spline.h"


class get_roi {

public:

    /******************************
     ******边缘检测（prewitt)******
     ******************************/

    static int xPrewittGradient(cv::Mat IG, int x, int y);
    static int yPrewittGradient(cv::Mat IG, int x, int y);
    static cv::Mat Prewitt(cv::Mat& src);


    /******************************
     *******边缘检测（sobel)*******
     ******************************/

    static int xSobelGradient(cv::Mat IG, int x, int y);
    static int ySobelGradient(cv::Mat IG, int x, int y);
    static cv::Mat Sobel(cv::Mat& src);
    

    /******************************
     ***********填充孔洞***********
     ******************************/

    static cv::Mat imfill(cv::Mat cop, int n);


    /******************************
     *********去除小连通域*********
     ******************************/

    static void RemoveSmallRegion(cv::Mat& Src, cv::Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode);


    /******************************
     *******凸包上下边界拟合*******
     ******************************/

    static cv::Mat poly_fit(cv::Mat& src, std::vector<cv::Point>& up, std::vector<cv::Point>& down);


    /******************************
     *******拉普拉斯边缘检测*******
     ******************************/

    static cv::Mat laplus(int kernel_size);


    /******************************
     ********多项式拟合算法********
     ******************************/

    static bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);


    /******************************
     *********B样条拟合边界********
     ******************************/

    static void BSpline_end(cv::Mat& src_convexhull, cv::Mat& dst_convexhull, cv::Mat& dst_bgr, cv::Mat& ROI, int& b, int& g, int& r);


    /******************************
     ********寻找最大连通域********
     ******************************/

    static void LargestConnecttedComponent(cv::Mat srcImage, cv::Mat& dstImage);


    /******************************
     ***********凸包算法***********
     ******************************/

    static cv::Mat convex_hull(cv::Mat& image);


    /******************************
     *******ROI上下边界映射********
     ******************************/

    static void poly_fit1(cv::Mat& src, cv::Mat& dst);


};