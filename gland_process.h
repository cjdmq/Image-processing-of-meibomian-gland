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
#include"get_roi.h"


class glandprocess
{
public:

    /******************************
     ***********ЖЈЯђЫузг***********
     ******************************/

    static void orientation_operator(cv::Mat& src, cv::Mat& dst, double& a, double& b);
};