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
     ******ά���˲�******
     ********************/

    /*bRow��bCol Mask����ʼ���кţ�winRow��winCol Mask�Ĵ�С*/
    static cv::Mat ReadMask(cv::Mat SourceImg, int bRow, int bCol, int winRow, int winCol);
    static int InterplotWallisParameter(float* grid, int gridRow, int gridCol, float* value, int x, int y);
    /*Mask����Wallis�˲������Ӱ������winRow��winColӰ�������С��r0��r1 Wallis�˲�ϵ��*/
    static int CalWallisParameter(cv::Mat Mask, int winRow, int winCol, float* r0, float* r1, float B_Value, float C_Value, float meanValue, float sigmaValue);
    static void Wallisfilter(cv::Mat SourceImg, cv::Mat Result, float B_Value, float C_Value, float sigmaValue, float meanValue);
    
    /********************
     ******CLAHE��ǿ*****
     ********************/

    static void color_transfer_with_spilt(cv::Mat& input, std::vector<cv::Mat>& chls);
    static void color_retransfer_with_merge(cv::Mat& output, std::vector<cv::Mat>& chls);
    static cv::Mat clahe_deal(cv::Mat& src);

   
};