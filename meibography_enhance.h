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


#define gridWIndowSize 20
#define filterWindowSize 70



class image_enhance {

public:

    /********************
     ******维纳滤波******
     ********************/

    /*bRow、bCol Mask的起始行列号；winRow、winCol Mask的大小*/
    static cv::Mat ReadMask(cv::Mat SourceImg, int bRow, int bCol, int winRow, int winCol);
    static int InterplotWallisParameter(float* grid, int gridRow, int gridCol, float* value, int x, int y);
    /*Mask进行Wallis滤波处理的影像区域；winRow、winCol影像区域大小；r0、r1 Wallis滤波系数*/
    static int CalWallisParameter(cv::Mat Mask, int winRow, int winCol, float* r0, float* r1, float B_Value, float C_Value, float meanValue, float sigmaValue);
    static void Wallisfilter(cv::Mat SourceImg, cv::Mat Result, float B_Value, float C_Value, float sigmaValue, float meanValue);
    
    /********************
     ******CLAHE增强*****
     ********************/

    static void color_transfer_with_spilt(cv::Mat& input, std::vector<cv::Mat>& chls);
    static void color_retransfer_with_merge(cv::Mat& output, std::vector<cv::Mat>& chls);
    static cv::Mat clahe_deal(cv::Mat& src);

   
};