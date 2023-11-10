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

    //Ƶ�����˲�
    static cv::Mat frequency_filter(cv::Mat& scr, cv::Mat& blur);

    //*****************�����ͨ�˲���***********************
    static cv::Mat ideal_low_kernel(cv::Mat& scr, float sigma);

    //�����ͨ�˲���
    static cv::Mat ideal_low_pass_filter(cv::Mat& src, float sigma);


    static cv::Mat butterworth_low_kernel(cv::Mat& scr, float sigma, int n);

    //������˹��ͨ�˲���
    static cv::Mat butterworth_low_paass_filter(cv::Mat& src, float d0, int n);

    static cv::Mat gaussian_low_pass_kernel(cv::Mat scr, float sigma);

    //��˹��ͨ
    static cv::Mat gaussian_low_pass_filter(cv::Mat& src, float d0);

    static cv::Mat ideal_high_kernel(cv::Mat& scr, float sigma);

    //�����ͨ�˲���
    static cv::Mat ideal_high_pass_filter(cv::Mat& src, float sigma);

    static cv::Mat butterworth_high_kernel(cv::Mat& scr, float sigma, int n);

    //������˹��ͨ�˲���
    static cv::Mat butterworth_high_paass_filter(cv::Mat& src, float d0, int n);

    static cv::Mat gaussian_high_pass_kernel(cv::Mat scr, float sigma);

    //��˹��ͨ
    static cv::Mat gaussian_high_pass_filter(cv::Mat& src, float d0);
};