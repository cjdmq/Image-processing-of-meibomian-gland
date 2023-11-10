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


class gaussianfilter {
public:
    static cv::Mat image_make_border(cv::Mat& src);

    //频率域滤波
    static cv::Mat frequency_filter(cv::Mat& scr, cv::Mat& blur);

    //*****************理想低通滤波器***********************
    static cv::Mat ideal_low_kernel(cv::Mat& scr, float sigma);

    //理想低通滤波器
    static cv::Mat ideal_low_pass_filter(cv::Mat& src, float sigma);


    static cv::Mat butterworth_low_kernel(cv::Mat& scr, float sigma, int n);

    //巴特沃斯低通滤波器
    static cv::Mat butterworth_low_paass_filter(cv::Mat& src, float d0, int n);

    static cv::Mat gaussian_low_pass_kernel(cv::Mat scr, float sigma);

    //高斯低通
    static cv::Mat gaussian_low_pass_filter(cv::Mat& src, float d0);

    static cv::Mat ideal_high_kernel(cv::Mat& scr, float sigma);

    //理想高通滤波器
    static cv::Mat ideal_high_pass_filter(cv::Mat& src, float sigma);

    static cv::Mat butterworth_high_kernel(cv::Mat& scr, float sigma, int n);

    //巴特沃斯高通滤波器
    static cv::Mat butterworth_high_paass_filter(cv::Mat& src, float d0, int n);

    static cv::Mat gaussian_high_pass_kernel(cv::Mat scr, float sigma);

    //高斯高通
    static cv::Mat gaussian_high_pass_filter(cv::Mat& src, float d0);
};