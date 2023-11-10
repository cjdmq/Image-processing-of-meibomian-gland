
#include"meibography_enhance.h"


/*bRow、bCol Mask的起始行列号；winRow、winCol Mask的大小*/
cv::Mat image_enhance::ReadMask(cv::Mat SourceImg, int bRow, int bCol, int winRow, int winCol)
{
    cv::Mat Mask(SourceImg, cv::Rect(bCol, bRow, winCol, winRow));
    return Mask;
}
/*grid计算出的r0或r1格网；gridRow、gridCol格网大小；*value插值结果；x、y需要进行插值计算的位置；*/
int	image_enhance::InterplotWallisParameter(float* grid, int gridRow, int gridCol, float* value, int x, int y)
{
    int      grid_r, grid_c;
    float    X, Y, L, Z00, Z10, Z01, Z11;
    grid_r = y / gridWIndowSize;
    grid_c = x / gridWIndowSize;
    if (grid_r < 0 || grid_c < 0 || grid_r >= gridRow - 1 || grid_c >= gridCol - 1)
    {
        if (grid_r <= 0)         grid_r = 0;
        if (grid_c <= 0)         grid_c = 0;
        if (grid_r >= gridRow - 1) grid_r = gridRow - 1;
        if (grid_c >= gridCol - 1) grid_c = gridCol - 1;
        x = grid_c * gridWIndowSize;
        y = grid_r * gridWIndowSize;
    }
    Z00 = *(grid + grid_r * gridCol + grid_c);
    Z10 = *(grid + grid_r * gridCol + grid_c + 1);
    Z01 = *(grid + (grid_r + 1) * gridCol + grid_c);
    Z11 = *(grid + (grid_r + 1) * gridCol + grid_c + 1);
    X = float(x - grid_c * gridWIndowSize);
    Y = float(y - grid_r * gridWIndowSize);
    L = float(gridWIndowSize);
    *value = (1 - X / L) * (1 - Y / L) * Z00 + X / L * (1 - Y / L) * Z10 + Y / L * (1 - X / L) * Z01 + X * Y / L / L * Z11;
    return 1;
}
/*Mask进行Wallis滤波处理的影像区域；winRow、winCol影像区域大小；r0、r1 Wallis滤波系数*/
int	image_enhance::CalWallisParameter(cv::Mat Mask, int winRow, int winCol, float* r0, float* r1, float B_Value, float C_Value, float meanValue, float sigmaValue)
{
    float mean0(0), sigma0(0);
    for (int row = 0; row < winRow; ++row)
    {
        for (int col = 0; col < winCol; ++col)
        {
            mean0 += Mask.at<uchar>(row, col);
            sigma0 += Mask.at<uchar>(row, col) * Mask.at<uchar>(row, col);
        }
    }

    int sum = winRow * winCol;
    float mean, sigma;
    mean = mean0 / float(sum);
    sigma = sigma0 / float(sum - mean * mean);
    if (sigma < 0) sigma = 0;
    sigma = float(sqrt(sigma));

    if (sigma == 0.0f)
    {
        *r1 = 1.0;
        *r0 = B_Value * meanValue + (1.0f - B_Value - *r1) * mean;
    }
    else
    {
        *r1 = C_Value * sigmaValue / (C_Value * sigma + (1.0f - C_Value) * sigmaValue);
        *r0 = B_Value * meanValue + (1.0f - B_Value - *r1) * mean;
    }
    return 1;
}
void image_enhance::Wallisfilter(cv::Mat SourceImg, cv::Mat Result, float B_Value, float C_Value, float sigmaValue, float meanValue)
{
    int heights = SourceImg.rows;
    int widths = SourceImg.cols;
    int	  x, y, br, bc, gridRow, gridCol;
    float r0, r1, rc, gf, rmean0, rsigma0, rmean, rsigma, * gridR0, * gridR1;
    gridRow = (heights - filterWindowSize) / gridWIndowSize;
    gridCol = (widths - filterWindowSize) / gridWIndowSize;
    gridR0 = (float*)calloc((gridRow + 1) * (gridCol + 1), sizeof(float));
    gridR1 = (float*)calloc((gridRow + 1) * (gridCol + 1), sizeof(float));
    rmean0 = rsigma0 = 0.0;
    cv::Mat Mask;
    for (int row = 0; row < gridRow; ++row)
    {
        br = row * gridWIndowSize;
        for (int col = 0; col < gridCol; ++col)
        {
            bc = col * gridWIndowSize;
            Mask = ReadMask(SourceImg, br, bc, filterWindowSize, filterWindowSize);
            CalWallisParameter(Mask, filterWindowSize, filterWindowSize, &r0, &r1, B_Value, C_Value, meanValue, sigmaValue);
            *(gridR0 + row * gridCol + col) = r0;
            *(gridR1 + row * gridCol + col) = r1;
            gf = (float)Mask.at<uchar>((filterWindowSize + 1) / 2, (filterWindowSize + 1) / 2);
            rc = gf * r1 + r0;
            rmean0 += rc;
            rsigma0 += rc * rc;
        }
    }
    rc = (float)(gridRow * gridCol);
    rmean = rmean0 / rc;
    rsigma = rsigma0 / rc - rmean * rmean;
    if (rsigma < 0) rsigma = 0;
    rsigma = float(sqrt(rsigma));
    for (int row = 0; row < heights; ++row)
    {
        y = row - filterWindowSize / 2;
        for (int col = 0; col < widths; ++col)
        {
            x = col - filterWindowSize / 2;
            InterplotWallisParameter(gridR0, gridRow, gridCol, &r0, x, y);
            InterplotWallisParameter(gridR1, gridRow, gridCol, &r1, x, y);
            gf = (float)Result.at<uchar>(row, col);
            if (gf <= 3.0 || gf >= 252.0) continue;
            rc = (gf * r1 + r0 - rmean) * sigmaValue / rsigma + meanValue;
            if (rc >= 256) rc = 255.0;
            else if (rc < 0) rc = 0.0;
            Result.at<uchar>(row, col) = (unsigned char)rc;
        }
    }
}


void image_enhance::color_transfer_with_spilt(cv::Mat& input, std::vector<cv::Mat>& chls)
{
    cv::cvtColor(input, input, cv::COLOR_BGR2YCrCb);
    split(input, chls);
}
void image_enhance::color_retransfer_with_merge(cv::Mat& output, std::vector<cv::Mat>& chls)
{
    merge(chls, output);
    cv::cvtColor(output, output, cv::COLOR_YCrCb2BGR);
}
cv::Mat image_enhance::clahe_deal(cv::Mat& src)
{
    cv::Mat ycrcb = src.clone();
    std::vector<cv::Mat> channels;

    color_transfer_with_spilt(ycrcb, channels);

    cv::Mat clahe_img;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    // 直方图的柱子高度大于计算后的ClipLimit的部分被裁剪掉，然后将其平均分配给整张直方图
    // 从而提升整个图像
    clahe->setClipLimit(10.0);    // (int)(8.*(16*16)/256)
    clahe->setTilesGridSize(cv::Size(16, 16)); // 将图像分为16*16块
    clahe->apply(channels[0], clahe_img);
    channels[0].release();
    clahe_img.copyTo(channels[0]);
    color_retransfer_with_merge(ycrcb, channels);
    return ycrcb;
}