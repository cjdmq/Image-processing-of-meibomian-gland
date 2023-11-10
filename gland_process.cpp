#include"gland_process.h"


/******************************
 ***********定向算子***********
 ******************************/

void glandprocess::orientation_operator(cv::Mat& src, cv::Mat& dst, double& a, double& b)
{
    cv::Mat smallest_rect = src.clone();
    cv::Mat resultImage = src.clone();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarcy;
    findContours(smallest_rect, contours, hierarcy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    std::vector<cv::RotatedRect> box(contours.size()); //最小外接矩形
    cv::Point2f rect[4];
    std::vector<cv::Mat>mask_presubtract(contours.size());
    std::vector<cv::Mat>masks(contours.size());
    std::vector<cv::Mat>maskss(contours.size());
    std::vector<cv::Mat>rezult_image(contours.size());
    std::vector<cv::Mat>gland_lagest(contours.size());
    /*char ad[128] = { 0 };
    int dx = 0;*/
    for (int i = 0; i < contours.size(); i++)
    {
        box[i] = cv::minAreaRect(cv::Mat(contours[i]));
        box[i].points(rect);          //最小外接矩形的4个端点
        double ellipseangle1 = 0;
        cv::Point2f p2[4];
        box[i].points(p2);
        //std::cout << p2[0] << std::endl;
        if (box[i].size.height <= box[i].size.width)
        {
            //cv::ellipse(resultImage, box[i], cv::Scalar(0, 255, 255), 1, 8);
            ellipseangle1 = -box[i].angle;
        }
        else
        {
            ellipseangle1 = -box[i].angle + 90;
        }
        masks[i] = cv::Mat::zeros(resultImage.size(), CV_8UC1);
        for (int j = 0; j < 4; j++)
        {
            if (ellipseangle1 > a && ellipseangle1 < b)
            {
                cv::line(masks[i], p2[j], p2[(j + 1) % 4], cv::Scalar(255), 1, 8);
            }
        }
        mask_presubtract[i] = src.clone();
        maskss[i] = get_roi::imfill(masks[i], 300000);
        cv::bitwise_not(maskss[i], maskss[i]);
        if (((maskss[i].rows * maskss[i].cols) - cv::countNonZero(maskss[i])) > 100)
        {
            cv::subtract(src, maskss[i], rezult_image[i]);
            gland_lagest[i] = rezult_image[i].clone();
            get_roi::LargestConnecttedComponent(rezult_image[i], gland_lagest[i]);
            cv::add(gland_lagest[i], dst, dst);

        }

        /*sprintf_s(ad, "C:\\Users\\Administrator\\Desktop\\新建文件夹 (2)\\test%d.jpg", ++dx);
        imwrite(ad, maskss[i]);*/
    }
}