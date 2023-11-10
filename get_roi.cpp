#include"get_roi.h"

/******************************
 ******边缘检测（prewitt)******
 ******************************/

int get_roi::xPrewittGradient(cv::Mat IG, int x, int y)
{
    return IG.at<uchar>(y - 1, x - 1) +
        IG.at<uchar>(y, x - 1) +
        IG.at<uchar>(y + 1, x - 1) -
        IG.at<uchar>(y - 1, x + 1) -
        IG.at<uchar>(y, x + 1) -
        IG.at<uchar>(y + 1, x + 1);
}
int get_roi::yPrewittGradient(cv::Mat IG, int x, int y)
{
    return IG.at<uchar>(y - 1, x - 1) -
        IG.at<uchar>(y + 1, x - 1) +
        IG.at<uchar>(y - 1, x) -
        IG.at<uchar>(y + 1, x) +
        IG.at<uchar>(y - 1, x + 1) -
        IG.at<uchar>(y + 1, x + 1);
}
cv::Mat get_roi::Prewitt(cv::Mat& src)
{
    cv::Mat image_Prewitt = src.clone();
    int gpx, gpy, sump;
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++) {
            image_Prewitt.at<uchar>(y, x) = 0.0;
        }
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            gpx = xPrewittGradient(src, x, y);
            gpy = yPrewittGradient(src, x, y);
            sump = sqrt(powf(gpx, 2.0) + powf(gpy, 2.0));
            sump = sump > 255 ? 255 : sump;
            sump = sump < 0 ? 0 : sump;
            image_Prewitt.at<uchar>(y, x) = sump;
        }
    }
    return image_Prewitt;
}


/******************************
 *******边缘检测（sobel)*******
 ******************************/

int get_roi::xSobelGradient(cv::Mat IG, int x, int y)
{
    return -IG.at<uchar>(y - 1, x - 1) - 2 *
        IG.at<uchar>(y, x - 1) -
        IG.at<uchar>(y + 1, x - 1) +
        IG.at<uchar>(y - 1, x + 1) + 2 *
        IG.at<uchar>(y, x + 1) +
        IG.at<uchar>(y + 1, x + 1);
}
int get_roi::ySobelGradient(cv::Mat IG, int x, int y)
{
    return -IG.at<uchar>(y - 1, x - 1) +
        IG.at<uchar>(y + 1, x - 1) - 2 *
        IG.at<uchar>(y - 1, x) + 2 *
        IG.at<uchar>(y + 1, x) -
        IG.at<uchar>(y - 1, x + 1) +
        IG.at<uchar>(y + 1, x + 1);
}
cv::Mat get_roi::Sobel(cv::Mat& src)
{
    cv::Mat image_Prewitt = src.clone();
    int gpx, gpy, sump;
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++) {
            image_Prewitt.at<uchar>(y, x) = 0.0;
        }
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            gpx = xSobelGradient(src, x, y);
            gpy = ySobelGradient(src, x, y);
            sump = sqrt(powf(gpx, 2.0) + powf(gpy, 2.0));
            sump = sump > 255 ? 255 : sump;
            sump = sump < 0 ? 0 : sump;
            image_Prewitt.at<uchar>(y, x) = sump;
        }
    }
    return image_Prewitt;
}


/******************************
 ***********填充孔洞***********
 ******************************/

cv::Mat get_roi::imfill(cv::Mat cop, int n)
{
    cv::Mat data = ~cop;
    cv::Mat labels, stats, centroids;
    connectedComponentsWithStats(data, labels, stats, centroids, 4, CV_16U);
    int regions_count = stats.rows - 1;
    int regions_size, regions_x1, regions_y1, regions_x2, regions_y2;

    for (int i = 1; i <= regions_count; i++)
    {
        regions_size = stats.ptr<int>(i)[4];
        if (regions_size < n)
        {
            regions_x1 = stats.ptr<int>(i)[0];
            regions_y1 = stats.ptr<int>(i)[1];
            regions_x2 = regions_x1 + stats.ptr<int>(i)[2];
            regions_y2 = regions_y1 + stats.ptr<int>(i)[3];

            for (int j = regions_y1; j < regions_y2; j++)
            {
                for (int k = regions_x1; k < regions_x2; k++)
                {
                    if (labels.ptr<ushort>(j)[k] == i)
                        data.ptr<uchar>(j)[k] = 0;
                }
            }
        }
    }
    data = ~data;
    return data;
}


/******************************
 *********去除小连通域*********
 ******************************/

void get_roi::RemoveSmallRegion(cv::Mat& Src, cv::Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
    int RemoveCount = 0;
    //新建一幅标签图像初始化为0像素点，为了记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查   
    //初始化的图像全部为0，未检查  
    cv::Mat PointLabel = cv::Mat::zeros(Src.size(), CV_8UC1);
    if (CheckMode == 1)//去除小连通区域的白色点  
    {
        //cout << "去除小连通域.";
        for (int i = 0; i < Src.rows; i++)
        {
            for (int j = 0; j < Src.cols; j++)
            {
                if (Src.at<uchar>(i, j) < 10)
                {
                    PointLabel.at<uchar>(i, j) = 3;//将背景黑色点标记为合格，像素为3  
                }
            }
        }
    }
    else//去除孔洞，黑色点像素  
    {
        //cout << "去除孔洞";
        for (int i = 0; i < Src.rows; i++)
        {
            for (int j = 0; j < Src.cols; j++)
            {
                if (Src.at<uchar>(i, j) > 10)
                {
                    PointLabel.at<uchar>(i, j) = 3;//如果原图是白色区域，标记为合格，像素为3  
                }
            }
        }
    }
    std::vector<cv::Point2i>NeihborPos;//将邻域压进容器  
    NeihborPos.push_back(cv::Point2i(-1, 0));
    NeihborPos.push_back(cv::Point2i(1, 0));
    NeihborPos.push_back(cv::Point2i(0, -1));
    NeihborPos.push_back(cv::Point2i(0, 1));
    if (NeihborMode == 1)
    {
        //cout << "Neighbor mode: 8邻域." << endl;
        NeihborPos.push_back(cv::Point2i(-1, -1));
        NeihborPos.push_back(cv::Point2i(-1, 1));
        NeihborPos.push_back(cv::Point2i(1, -1));
        NeihborPos.push_back(cv::Point2i(1, 1));
    }
    /*else
    cout << "Neighbor mode: 4邻域." << endl;*/
    int NeihborCount = 4 + 4 * NeihborMode;
    int CurrX = 0, CurrY = 0;
    //开始检测  
    for (int i = 0; i < Src.rows; i++)
    {
        for (int j = 0; j < Src.cols; j++)
        {
            if (PointLabel.at<uchar>(i, j) == 0)//标签图像像素点为0，表示还未检查的不合格点  
            {   //开始检查  
                std::vector<cv::Point2i>GrowBuffer;//记录检查像素点的个数  
                GrowBuffer.push_back(cv::Point2i(j, i));
                PointLabel.at<uchar>(i, j) = 1;//标记为正在检查  
                int CheckResult = 0;
                for (int z = 0; z < GrowBuffer.size(); z++)
                {
                    for (int q = 0; q < NeihborCount; q++)
                    {
                        CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
                        CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
                        if (CurrX >= 0 && CurrX < Src.cols && CurrY >= 0 && CurrY < Src.rows)  //防止越界    
                        {
                            if (PointLabel.at<uchar>(CurrY, CurrX) == 0)
                            {
                                GrowBuffer.push_back(cv::Point2i(CurrX, CurrY));  //邻域点加入buffer    
                                PointLabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查    
                            }
                        }
                    }
                }
                if (GrowBuffer.size() > AreaLimit) //判断结果（是否超出限定的大小），1为未超出，2为超出    
                    CheckResult = 2;
                else
                {
                    CheckResult = 1;
                    RemoveCount++;//记录有多少区域被去除  
                }
                for (int z = 0; z < GrowBuffer.size(); z++)
                {
                    CurrX = GrowBuffer.at(z).x;
                    CurrY = GrowBuffer.at(z).y;
                    PointLabel.at<uchar>(CurrY, CurrX) += CheckResult;//标记不合格的像素点，像素值为2  
                }
                //********结束该点处的检查**********    
            }
        }
    }
    CheckMode = 255 * (1 - CheckMode);
    //开始反转面积过小的区域    
    for (int i = 0; i < Src.rows; ++i)
    {
        for (int j = 0; j < Src.cols; ++j)
        {
            if (PointLabel.at<uchar>(i, j) == 2)
            {
                Dst.at<uchar>(i, j) = CheckMode;
            }
            else if (PointLabel.at<uchar>(i, j) == 3)
            {
                Dst.at<uchar>(i, j) = Src.at<uchar>(i, j);
            }
        }
    }
    //cout << RemoveCount << " objects removed." << endl;
}


/******************************
 *******凸包上下边界拟合*******
 ******************************/

cv::Mat get_roi::poly_fit(cv::Mat& src, std::vector<cv::Point>& up, std::vector<cv::Point>& down)
{
    cv::Mat src1 = src.clone();
    std::vector<std::vector<cv::Point>>points_group;//将每一列的白色像素的坐标存起来
    for (int j = 0; j < src1.cols; j++)
    {
        std::vector<cv::Point>points;
        for (int i = 0; i < src1.rows; i++)
        {
            uchar* data = src1.ptr<uchar>(i);
            //points.push_back(ICH_RemoveSmallRegions_rows_ptr[i]);

            if (data[j] == 255)
            {
                cv::Point point1(j, i);
                points.push_back(point1);
            }
        }
        points_group.push_back(points);
        //std::cout << points << std::endl;
    }
    std::vector<cv::Point>points_first;
    std::vector<cv::Point>points_end;
    for (int i = 0; i < points_group.size(); i++)
    {
        if (!points_group[i].empty())
        {
            //std::cout << points_group[i] << std::endl;
            cv::Point first = points_group[i].front();
            cv::Point end = points_group[i].back();
            points_first.push_back(first);
            points_end.push_back(end);
            /*std::cout << first << std::endl;
            std::cout << end << std::endl;*/
        }
    }
    //std::cout << points_first << std::endl;
    //std::cout << points_end << std::endl;
    cv::Mat drawbackground = src1.clone();
    cv::cvtColor(src1, drawbackground, cv::COLOR_GRAY2BGR);
    polylines(drawbackground, points_first, false, cv::Scalar(0, 255, 255), 4, cv::LINE_8);
    //imshow("sajdkajdk", drawbackground);
    for (std::vector<cv::Point>::const_iterator nIterator = points_first.begin(); nIterator != points_first.end(); nIterator++)
    {
        up.push_back(*nIterator);
    }
    polylines(drawbackground, points_end, false, cv::Scalar(255, 0, 255), 4, cv::LINE_8);
    for (std::vector<cv::Point>::const_iterator nIterator = points_end.begin(); nIterator != points_end.end(); nIterator++)
    {
        down.push_back(*nIterator);
    }
    return drawbackground;
}


/******************************
 *******ROI上下边界映射********
 ******************************/

void get_roi::poly_fit1(cv::Mat& src, cv::Mat& dst)
{
    cv::Mat src1 = src.clone();
    std::vector<cv::Point> up;
    std::vector<cv::Point> down;
    std::vector<std::vector<cv::Point>>points_group;//将每一列的白色像素的坐标存起来
    for (int j = 0; j < src1.cols; j++)
    {
        std::vector<cv::Point>points;
        for (int i = 0; i < src1.rows; i++)
        {
            uchar* data = src1.ptr<uchar>(i);
            //points.push_back(ICH_RemoveSmallRegions_rows_ptr[i]);

            if (data[j] == 255)
            {
                cv::Point point1(j, i);
                points.push_back(point1);
            }
        }
        points_group.push_back(points);
        //std::cout << points << std::endl;
    }
    std::vector<cv::Point>points_first;
    std::vector<cv::Point>points_end;
    for (int i = 0; i < points_group.size(); i++)
    {
        if (!points_group[i].empty())
        {
            //std::cout << points_group[i] << std::endl;
            cv::Point first = points_group[i].front();
            cv::Point end = points_group[i].back();
            points_first.push_back(first);
            points_end.push_back(end);
            /*std::cout << first << std::endl;
            std::cout << end << std::endl;*/
        }
    }
    //std::cout << points_first << std::endl;
    //std::cout << points_end << std::endl;
    
    polylines(dst, points_first, false, cv::Scalar(0, 255, 0), 8, cv::LINE_AA);
    //cv::imshow("sajdkajdk", dst);
    for (std::vector<cv::Point>::const_iterator nIterator = points_first.begin(); nIterator != points_first.end(); nIterator++)
    {
        up.push_back(*nIterator);
    }
    polylines(dst, points_end, false, cv::Scalar(0, 255, 0), 8, cv::LINE_AA);
    for (std::vector<cv::Point>::const_iterator nIterator = points_end.begin(); nIterator != points_end.end(); nIterator++)
    {
        down.push_back(*nIterator);
    }
}


/******************************
 *******拉普拉斯边缘检测*******
 ******************************/

cv::Mat get_roi::laplus(int kernel_size)
{
    cv::Mat kernel = (cv::Mat_<float>(kernel_size, kernel_size));
    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
            if (i == (kernel_size - 1) / 2 && j == (kernel_size - 1) / 2)
            {
                kernel.ptr<float>(i)[j] = kernel_size * kernel_size - 1;
            }
            else
            {
                kernel.ptr<float>(i)[j] = -1;
            }
    }
    return kernel;
}


/******************************
 ********多项式拟合算法********
 ******************************/

bool get_roi::polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
    //Number of key points
    int N = key_point.size();

    //构造矩阵X
    cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            for (int k = 0; k < N; k++)
            {
                X.at<double>(i, j) = X.at<double>(i, j) + pow(key_point[k].x, i + j);
            }
        }
    }
    //构造矩阵Y
    cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int k = 0; k < N; k++)
        {
            Y.at<double>(i, 0) = Y.at<double>(i, 0) + pow(key_point[k].x, i) * key_point[k].y;
        }
    }

    A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    //求解矩阵A
    solve(X, Y, A, cv::DECOMP_LU);
    return true;
}


/******************************
 *********B样条拟合边界********
 ******************************/

void get_roi::BSpline_end(cv::Mat& src_convexhull, cv::Mat& dst_convexhull, cv::Mat& dst_bgr, cv::Mat& ROI, int& b, int& g, int& r)
{

    std::vector<cv::Point> up;
    std::vector<cv::Point> down;

    dst_convexhull = poly_fit(src_convexhull, up, down);

    std::vector<cv::Point> up_result;
    std::vector<cv::Point> down_result;
    BSpline bs1(4, 100, up, up_result);
    //std::cout << up_result << std::endl;
    for (int i = 0; i < 100 - 1; ++i)
    {
        int j = (i + 1) % 100;
        cv::line(dst_convexhull, cv::Point2d(bs1[i].x, bs1[i].y), cv::Point2d(bs1[i + 1].x, bs1[i + 1].y), CV_RGB(255, 0, 0), 2, 8, 0);
    }
    BSpline bs2(4, 100, down, down_result);
    //std::cout << down_result << std::endl;
    for (int i = 0; i < 100 - 1; ++i)
    {
        int j = (i + 1) % 100;
        cv::line(dst_convexhull, cv::Point2d(bs2[i].x, bs2[i].y), cv::Point2d(bs2[i + 1].x, bs2[i + 1].y), CV_RGB(0, 0, 255), 2, 8, 0);
    }

    cv::Mat A;
    cv::Mat B;
    /*cout << "h(q.rows)：" << q.rows << "  " << "w(q.cols)：" << q.cols << endl;*/

    polynomial_curve_fit(up_result, 3, A);
    std::vector<cv::Point> points_fitted_up;
    for (int x = 0; x < dst_convexhull.cols; x++)
    {
        double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x + A.at<double>(2, 0) * pow(x, 2) + A.at<double>(3, 0) * pow(x, 3);
        points_fitted_up.push_back(cv::Point(x, y));
    }

    polylines(dst_convexhull, points_fitted_up, false, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    //cv::imshow("sdadasad", dst_convexhull);
    //polylines(dst_bgr, points_fitted_up, false, cv::Scalar(b, g, r), 6, cv::LINE_AA);

    polynomial_curve_fit(down_result, 3, B);
    std::vector<cv::Point> points_fitted_down;
    for (int x = 0; x < dst_convexhull.cols; x++)
    {
        double y = B.at<double>(0, 0) + B.at<double>(1, 0) * x + B.at<double>(2, 0) * pow(x, 2) + B.at<double>(3, 0) * pow(x, 3);
        points_fitted_down.push_back(cv::Point(x, y));
    }
    polylines(dst_convexhull, points_fitted_down, false, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    //polylines(dst_bgr, points_fitted_down, false, cv::Scalar(b, g, r), 6, cv::LINE_AA);
    //将轮廓内区域提取出来
    ROI = cv::Mat::zeros(src_convexhull.size(), CV_8UC3);
    for (int i = 0; i <= ROI.cols; i++)
    {
        for (int j = 0; j < ROI.rows; j++)
        {
            if (j >= points_fitted_up[i].y && j <= points_fitted_down[i].y)
            {
                ROI.at<cv::Vec3b>(j, i)[0] = 255;
                ROI.at<cv::Vec3b>(j, i)[1] = 255;
                ROI.at<cv::Vec3b>(j, i)[2] = 255;
            }
        }
    }
    cv::cvtColor(ROI, ROI, cv::COLOR_BGR2GRAY);
}


/******************************
 ********寻找最大连通域********
 ******************************/

void get_roi::LargestConnecttedComponent(cv::Mat srcImage, cv::Mat& dstImage)
{
    cv::Mat temp;
    cv::Mat labels;
    srcImage.copyTo(temp);

    //1. 标记连通域
    int n_comps = connectedComponents(temp, labels, 4, CV_16U);
    std::vector<int> histogram_of_labels;
    for (int i = 0; i < n_comps; i++)//初始化labels的个数为0
    {
        histogram_of_labels.push_back(0);
    }

    int rows = labels.rows;
    int cols = labels.cols;
    for (int row = 0; row < rows; row++) //计算每个labels的个数
    {
        for (int col = 0; col < cols; col++)
        {
            histogram_of_labels.at(labels.at<unsigned short>(row, col)) += 1;
        }
    }
    histogram_of_labels.at(0) = 0; //将背景的labels个数设置为0

    //2. 计算最大的连通域labels索引
    int maximum = 0;
    int max_idx = 0;
    for (int i = 0; i < n_comps; i++)
    {
        if (histogram_of_labels.at(i) > maximum)
        {
            maximum = histogram_of_labels.at(i);
            max_idx = i;
        }
    }

    //3. 将最大连通域标记为1
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            if (labels.at<unsigned short>(row, col) == max_idx)
            {
                labels.at<unsigned short>(row, col) = 255;
            }
            else
            {
                labels.at<unsigned short>(row, col) = 0;
            }
        }
    }

    //4. 将图像更改为CV_8U格式
    labels.convertTo(dstImage, CV_8U);
}


/******************************
 ***********凸包算法***********
 ******************************/

cv::Mat get_roi::convex_hull(cv::Mat& image)
{
    //凸包算子
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(image, contours, hierarchy, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    //绘制轮廓
    cv::Mat contours_img(image.size(), CV_8U, cv::Scalar(0));
    drawContours(contours_img, contours, -1, cv::Scalar(255), 1);

    //凸包检测
    std::vector<std::vector<cv::Point> > pointHull(contours.size());
    std::vector <std::vector<int> > intHull(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        // 输出结果为Point类型的凸包检测
        convexHull(cv::Mat(contours[i]), pointHull[i], false);
        // 输出结果为int类型的凸包检测
        convexHull(cv::Mat(contours[i]), intHull[i], false);
    }
    //绘制凸包
    cv::Mat ICH = cv::Mat::zeros(image.size(), CV_8UC1);
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(255);
        drawContours(ICH, pointHull, i, color, -1, cv::LINE_8, std::vector<cv::Vec4i>(), 0, cv::Point());
    }
    return ICH;
}