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
     ******��Ե��⣨prewitt)******
     ******************************/

    static int xPrewittGradient(cv::Mat IG, int x, int y);
    static int yPrewittGradient(cv::Mat IG, int x, int y);
    static cv::Mat Prewitt(cv::Mat& src);


    /******************************
     *******��Ե��⣨sobel)*******
     ******************************/

    static int xSobelGradient(cv::Mat IG, int x, int y);
    static int ySobelGradient(cv::Mat IG, int x, int y);
    static cv::Mat Sobel(cv::Mat& src);
    

    /******************************
     ***********���׶�***********
     ******************************/

    static cv::Mat imfill(cv::Mat cop, int n);


    /******************************
     *********ȥ��С��ͨ��*********
     ******************************/

    static void RemoveSmallRegion(cv::Mat& Src, cv::Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode);


    /******************************
     *******͹�����±߽����*******
     ******************************/

    static cv::Mat poly_fit(cv::Mat& src, std::vector<cv::Point>& up, std::vector<cv::Point>& down);


    /******************************
     *******������˹��Ե���*******
     ******************************/

    static cv::Mat laplus(int kernel_size);


    /******************************
     ********����ʽ����㷨********
     ******************************/

    static bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);


    /******************************
     *********B������ϱ߽�********
     ******************************/

    static void BSpline_end(cv::Mat& src_convexhull, cv::Mat& dst_convexhull, cv::Mat& dst_bgr, cv::Mat& ROI, int& b, int& g, int& r);


    /******************************
     ********Ѱ�������ͨ��********
     ******************************/

    static void LargestConnecttedComponent(cv::Mat srcImage, cv::Mat& dstImage);


    /******************************
     ***********͹���㷨***********
     ******************************/

    static cv::Mat convex_hull(cv::Mat& image);


    /******************************
     *******ROI���±߽�ӳ��********
     ******************************/

    static void poly_fit1(cv::Mat& src, cv::Mat& dst);


};