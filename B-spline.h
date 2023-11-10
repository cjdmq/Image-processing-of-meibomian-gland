#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<string>
#include<vector>
#include<time.h>
#include<algorithm>
#include<stdio.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<stdint.h>
#include<opencv2/core.hpp>
#include<numeric>
#include<iomanip>


class BSpline {
public:
    BSpline();
    /* BSpline(const BSpline &bs){} */
    BSpline(int n, int resolution, std::vector<cv::Point> control_points, std::vector<cv::Point>& result);
    ~BSpline();
    /*
     * void release(){
     *     /\* basic_mat_.release(); *\/
     *     /\*
     *      * if (curve_ != NULL) delete [] curve_;
     *      * if (tangent_ != NULL) delete [] tangent_;
     *      *\/
     * }
     */

     /* void clear(){delete [] basic_mat_; delete [] curve_; delete [] tangent_;} */
    inline cv::Point& operator[](const size_t index)
    {
        return curve_[index];
    }

    inline const cv::Point& operator[](const size_t index) const
    {
        return curve_[index];
    }

    inline cv::Point& dt(const size_t index)
    {
        return tangent_[index];
    }
    cv::Mat basic_mat_;
private:
    void computeKnots();
    void computePoint(std::vector<cv::Point> control,
        cv::Point* p,
        cv::Point* tangent,
        double* mat_ptr,
        double v,
        int degree);
    double basic(int k, int t, double v);
    double basic(int i, int degree, double t, double* bp);
    std::vector<int> knots;
    cv::Point* curve_;
    cv::Point* tangent_;
};