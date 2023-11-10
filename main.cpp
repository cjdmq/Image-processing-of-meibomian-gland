#include"meibography_enhance.h"
#include"get_roi.h"
#include"gaussian_fliter.h"
#include"binarizate.h"



int main(int argc, char** argv)

{
    double t0 = (double)cv::getTickCount();//计时函数

    cv::Mat src = cv::imread("E:\\desktop\\睑板腺采集图像\\上眼睑\\5.PNG");

    if (!src.data)
    {
        std::cout << "没有检测到图片，请检查文件路径是否正确！" << std::endl;
        return -1;
    }

    //将原始图像转换成灰度图像
    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    //将灰度图像复制一次作为输出图像传进维纳滤波函数中
    cv::Mat src_wallisfilter = src_gray.clone();
    float B_Value1 = 0.5;   // 影像反差扩展常数
    float C_Value1 = 0.5;   // 影像亮度系数
    float meanValue1 = 100; // 均值目标值
    float sigmaValue1 = 40;// 方差目标值
    image_enhance::Wallisfilter(src_gray, src_wallisfilter, B_Value1, C_Value1, meanValue1, sigmaValue1);

    cv::Mat src_wallisfilter_bgr;
    cv::cvtColor(src_wallisfilter, src_wallisfilter_bgr, cv::COLOR_GRAY2BGR);

    int w = src_wallisfilter.cols;
    int h = src_wallisfilter.rows;

    //std::cout << w << "   " << h << std::endl;


    cv::Point p1(1.8 * w / 3, 0);
    cv::Point p2(w, 0);
    cv::Point p3(w, 1.5 * h / 3);

    std::vector<cv::Point>pts1;

    pts1.push_back(p1);
    pts1.push_back(p2);
    pts1.push_back(p3);

    std::vector<std::vector<cv::Point>>contours;
    contours.push_back(pts1);

    drawContours(src_wallisfilter, contours, -1, cv::Scalar(0, 0, 0), -1);


    //cv::Mat image_prewitt = get_roi::Prewitt(src_wallisfilter);




    cv::Mat image_sobel = get_roi::Sobel(src_wallisfilter);

    cv::threshold(image_sobel, image_sobel, 210, 255, cv::THRESH_BINARY);

    cv::Mat image_sobel1 = image_sobel.clone();
    get_roi::RemoveSmallRegion(image_sobel, image_sobel1, 50, 1, 1);


    cv::Mat image_sobel1_dilate = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::dilate(image_sobel1, image_sobel1, image_sobel1_dilate);

    cv::Mat image_fill1 = get_roi::imfill(image_sobel1, 200000);

    cv::Mat image_rezult;

    cv::subtract(src_wallisfilter, image_fill1, image_rezult);


    cv::Mat rezult_thres;

    cv::threshold(image_rezult, rezult_thres, 65, 255, cv::THRESH_BINARY);

    cv::Mat rezult_lagest = rezult_thres.clone();

    get_roi::LargestConnecttedComponent(rezult_thres, rezult_lagest);

    cv::Mat rezult_lagest_fill = get_roi::imfill(rezult_lagest, 200000);

    cv::Mat rezult_lagest_fill_dilate_ELE = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(50, 30));

    cv::dilate(rezult_lagest_fill, rezult_lagest_fill, rezult_lagest_fill_dilate_ELE, cv::Point(-1, -1));

    cv::Mat rezult_lagest_convell = get_roi::convex_hull(rezult_lagest_fill);

    //cv::subtract(image_rezult, rezult_lagest_convell, rezult_lagest_convell);


    cv::Mat dst_convexhull, dst_bgr, ROI1;
    int q = 0;
    int m = 0;
    int e = 255;
    get_roi::BSpline_end(rezult_lagest_convell, dst_convexhull, src_wallisfilter_bgr, ROI1, q, w, e);

    cv::Mat ROI = ROI1.clone();

    get_roi::LargestConnecttedComponent(ROI1, ROI);
    

    get_roi::poly_fit1(ROI, src_wallisfilter_bgr);

    //cv::imwrite("C:\\Users\\Administrator\\Desktop\\roi.png", src_wallisfilter_bgr);

    cv::Mat gaussion_low = gaussianfilter::gaussian_low_pass_filter(image_rezult, 80);
    gaussion_low = gaussion_low(cv::Rect(0, 0, image_rezult.cols, image_rezult.rows));
    /*cv::namedWindow("高斯低通", cv::WINDOW_FREERATIO);
    imshow("高斯低通", gaussion_low);*/

    cv::Mat gaussion_high = gaussianfilter::gaussian_high_pass_filter(image_rezult, 50);
    gaussion_high = gaussion_high(cv::Rect(0, 0, image_rezult.cols, image_rezult.rows));
    /*cv::namedWindow("高斯高通", cv::WINDOW_FREERATIO);
    imshow("高斯高通", gaussion_high);*/

    //std::cout << gaussion_high.channels() << std::endl;

    int blockSize = 201;
    int constValue = 1;
    //const int maxVal = 255;
    /* 自适应阈值算法
    0：ADAPTIVE_THRESH_MEAN_C
    1: ADAPTIVE_THRESH_GAUSSIAN_C
    阈值类型
    0: THRESH_BINARY
    1: THRESH_BINARY_INV */
    int adaptiveMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
    int thresholdType = 0;




    cv::Mat H_L;
    cv::subtract(gaussion_high, gaussion_low, H_L);

    cv::Mat L_H;
    cv::subtract(gaussion_low, gaussion_high, L_H);

    H_L.convertTo(H_L, CV_8U, 255.0);
    L_H.convertTo(L_H, CV_8U, 255.0);


    cv::Mat H_L_THRES = cv::Mat::zeros(H_L.size(), H_L.type());
   
    Binarizate::AdaptiveBinarizate(H_L, H_L_THRES, 255, adaptiveMethod, thresholdType, blockSize, constValue);


    cv::Mat L_H_THRES = cv::Mat::zeros(L_H.size(), L_H.type());
    Binarizate::AdaptiveBinarizate(L_H, L_H_THRES, 255, adaptiveMethod, thresholdType, blockSize, constValue);


    cv::namedWindow("输入图像", cv::WINDOW_FREERATIO);
    cv::imshow("输入图像", dst_convexhull);


    cv::namedWindow("维纳滤波增强", cv::WINDOW_FREERATIO);
    cv::imshow("维纳滤波增强", L_H_THRES);

   
    



    double t1 = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    std::cout << "本算法所运行的时间：" << t1 << "S" << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}