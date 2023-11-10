/** @file
* @brief 自动计算单通道图像二值化阈值
* @author ImageJ fh
* 由于VC编译器Max, Min函数可能会出现问题, 所以有关求最值函数全部自定义在cpp文件中
*/
#include"binarizate.h"


int Binarizate::OTSU(cv::Mat srcImage)
{
	int nCols = srcImage.cols;
	int nRows = srcImage.rows;
	int threshold = 0;
	// 初始化统计参数
	int nSumPix[256];
	float nProDis[256];
	for (int i = 0; i < 256; i++)
	{
		nSumPix[i] = 0;
		nProDis[i] = 0;
	}
	// 统计灰度级中每个像素在整幅图像中的个数 
	for (int i = 0; i < nCols; i++)
	{
		for (int j = 0; j < nRows; j++)
		{
			nSumPix[(int)srcImage.at<uchar>(i, j)]++;
		}
	}
	// 计算每个灰度级占图像中的概率分布
	for (int i = 0; i < 256; i++)
	{
		nProDis[i] = (float)nSumPix[i] / (nCols * nRows);
	}
	// 遍历灰度级[0,255],计算出最大类间方差下的阈值  
	float w0, w1, u0_temp, u1_temp, u0, u1, delta_temp;
	double delta_max = 0.0;
	for (int i = 0; i < 256; i++)
	{
		// 初始化相关参数
		w0 = w1 = u0_temp = u1_temp = u0 = u1 = delta_temp = 0;
		for (int j = 0; j < 256; j++)
		{
			//背景部分 
			if (j <= i)
			{
				// 当前i为分割阈值，第一类总的概率  
				w0 += nProDis[j];
				u0_temp += j * nProDis[j];
			}
			//前景部分   
			else
			{
				// 当前i为分割阈值，第一类总的概率
				w1 += nProDis[j];
				u1_temp += j * nProDis[j];
			}
		}
		// 分别计算各类的平均灰度 
		u0 = u0_temp / w0;
		u1 = u1_temp / w1;
		delta_temp = (float)(w0 * w1 * pow((u0 - u1), 2));
		// 依次找到最大类间方差下的阈值    
		if (delta_temp > delta_max)
		{
			delta_max = delta_temp;
			threshold = i;
		}
	}
	return threshold;
}

int Binarizate::OtsuBinarizate(cv::Mat srcImage, cv::Mat& dstImage, int maxVal, int threshType)
{
	if (!srcImage.data || srcImage.channels() != 1)
	{
		return 1;
	}
	// 初始化阈值参数
	int thresh = OTSU(srcImage);
	// 初始化阈值化处理的类型 
	/* 0: 二进制阈值 1: 反二进制阈值 2: 截断阈值
	3: 0阈值   4: 反0阈值*/
	//int threshType = 0;
	// 预设最大值
	//const int maxVal = 255;
	// 固定阈值化操作
	cv::threshold(srcImage, dstImage, thresh,
		maxVal, threshType);
	return 0;

}

int Binarizate::FixedBinarizate(cv::Mat srcImage, cv::Mat& dstImage, int thresh, int maxVal, int threshType)
{
	if (!srcImage.data || srcImage.channels() != 1)
	{
		return 1;
	}
	// 初始化阈值参数
	//int thresh = 130;
	// 初始化阈值化处理的类型 
	/* 0: 二进制阈值 1: 反二进制阈值 2: 截断阈值
	3: 0阈值   4: 反0阈值 8:大均法*/
	//int threshType = 0;
	// 预设最大值
	//const int maxVal = 255;
	// 固定阈值化操作
	cv::threshold(srcImage, dstImage, thresh,
		maxVal, threshType);
	return 0;
}

int Binarizate::AdaptiveBinarizate(cv::Mat srcImage, cv::Mat& dstImage, int maxVal, int adaptiveMethod, int thresholdType, int blockSize, int constValue)
{
	if (!srcImage.data || srcImage.channels() != 1)
	{
		return 1;
	}
	// 初始化自适应阈值参数
	//int blockSize = 5;
	//int constValue = 10;
	//const int maxVal = 255;
	/* 自适应阈值算法
	0：ADAPTIVE_THRESH_MEAN_C
	1: ADAPTIVE_THRESH_GAUSSIAN_C
	阈值类型
	0: THRESH_BINARY
	1: THRESH_BINARY_INV */
	//int adaptiveMethod = 0;
	//int thresholdType = 1;
	// 图像自适应阈值操作
	cv::adaptiveThreshold(srcImage, dstImage, maxVal, adaptiveMethod, thresholdType, blockSize, constValue);
	return 0;
}


/** @file
* @brief 自动计算单通道图像二值化阈值
* @author ImageJ fh
* 由于VC编译器Max, Min函数可能会出现问题, 所以有关求最值函数全部自定义在cpp文件中
*/

/*****************************************常用函数*********************************************/
static int CalcMaxValue(int a, int b)
{
    return (a > b) ? a : b;
}

static double CalcMaxValue(double a, double b)
{
    return (a > b) ? a : b;
}

static int CalcMinValue(int a, int b)
{
    return (a < b) ? a : b;
}

static double CalcMinValue(double a, double b)
{
    return (a < b) ? a : b;
}

void Binarizate::CalcHist(const cv::Mat& src, std::vector<int>& data)
{
    CV_Assert(CV_8UC1 == src.type());
    data.clear();

    // 初始化数组
    const int GRAY_LEVEL = 256;
    for (int i = 0; i < GRAY_LEVEL; ++i)
    {
        data.push_back(0);
    }

    // 统计每个灰度级及其对应的像素个数
    for (int y = 0; y < src.rows; ++y)
    {
        for (int x = 0; x < src.cols; ++x)
        {
            int value = src.at<uchar>(y, x);
            data.at(value) += 1;
        }
    }
}

int Binarizate::DefaultIsoData(const std::vector<int>& data)
{
    // This is the modified IsoData method used by the "Threshold" widget in "Default" mode
    int n = data.size();
    std::vector<int> data2(n, 0);
    int mode = 0, maxCount = 0;
    for (int i = 0; i < n; i++)
    {
        int count = data[i];
        data2[i] = data[i];
        if (data2[i] > maxCount)
        {
            maxCount = data2[i];
            mode = i;
        }
    }
    int maxCount2 = 0;
    for (int i = 0; i < n; i++)
    {
        if ((data2[i] > maxCount2) && (i != mode))
            maxCount2 = data2[i];
    }
    int hmax = maxCount;
    if ((hmax > (maxCount2 * 2)) && (maxCount2 != 0))
    {
        hmax = (int)(maxCount2 * 1.5);
        data2[mode] = hmax;
    }
    return IJIsoData(data2);

}

int Binarizate::IJIsoData(std::vector<int>& data)
{
    // This is the original ImageJ IsoData implementation, here for backward compatibility.
    int level;
    int maxValue = data.size() - 1;
    double result, sum1, sum2, sum3, sum4;
    int count0 = data[0];
    data[0] = 0; //set to zero so erased areas aren't included
    int countMax = data[maxValue];
    data[maxValue] = 0;
    int min = 0;
    while ((data[min] == 0) && (min < maxValue))
        min++;
    int max = maxValue;
    while ((data[max] == 0) && (max > 0))
        max--;
    if (min >= max)
    {
        data[0] = count0; data[maxValue] = countMax;
        level = data.size() / 2;
        return level;
    }
    int movingIndex = min;
    int inc = CalcMaxValue(max / 40, 1);
    do
    {
        sum1 = sum2 = sum3 = sum4 = 0.0;
        for (int i = min; i <= movingIndex; i++)
        {
            sum1 += (double)i * data[i];
            sum2 += data[i];
        }
        for (int i = (movingIndex + 1); i <= max; i++)
        {
            sum3 += (double)i * data[i];
            sum4 += data[i];
        }
        result = (sum1 / sum2 + sum3 / sum4) / 2.0;
        movingIndex++;
    } while ((movingIndex + 1) <= result && movingIndex < max - 1);
    data[0] = count0; data[maxValue] = countMax;
    level = (int)std::round(result);
    return level;
}

int Binarizate::Huang(const std::vector<int>& data)
{
    const int   GRAY_LEVELS = 256;
    int         threshold = -1;
    int         ih = 0;
    int         it = 0;
    int         first_bin = 0;
    int         last_bin = 0;
    double      sum_pix = 0.0;
    double      num_pix = 0.0;
    double      term = 0.0;
    double      ent = 0.0;      // entropy 
    double      min_ent = 0.0;  // min entropy 
    double      mu_x = 0.0;

    /* Determine the first non-zero bin */
    first_bin = 0;
    for (ih = 0; ih < GRAY_LEVELS; ih++)
    {
        if (data[ih] != 0)
        {
            first_bin = ih;
            break;
        }
    }

    /* Determine the last non-zero bin */
    last_bin = 255;
    for (ih = 255; ih >= first_bin; ih--)
    {
        if (data[ih] != 0)
        {
            last_bin = ih;
            break;
        }
    }
    term = 1.0 / (double)(last_bin - first_bin);
    std::vector<double> mu_0(GRAY_LEVELS, 0.0);
    sum_pix = num_pix = 0;
    for (ih = first_bin; ih < GRAY_LEVELS; ih++)
    {
        sum_pix += (double)ih * data[ih];
        num_pix += data[ih];
        /* NUM_PIX cannot be zero ! */
        mu_0[ih] = sum_pix / num_pix;
    }

    //double[] mu_1 = new double[256];
    std::vector<double> mu_1(GRAY_LEVELS, 0.0);
    sum_pix = num_pix = 0;
    for (ih = last_bin; ih > 0; ih--)
    {
        sum_pix += (double)ih * data[ih];
        num_pix += data[ih];
        /* NUM_PIX cannot be zero ! */
        mu_1[ih - 1] = sum_pix / (double)num_pix;
    }

    /* Determine the threshold that minimizes the fuzzy entropy */
    threshold = -1;
    min_ent = DBL_MAX;
    for (it = 0; it < GRAY_LEVELS; it++)
    {
        ent = 0.0;
        for (ih = 0; ih <= it; ih++)
        {
            /* Equation (4) in Ref. 1 */
            mu_x = 1.0 / (1.0 + term * std::abs(ih - mu_0[it]));
            if (!((mu_x < 1e-06) || (mu_x > 0.999999)))
            {
                /* Equation (6) & (8) in Ref. 1 */
                ent += data[ih] * (-mu_x * std::log(mu_x) - (1.0 - mu_x) * std::log(1.0 - mu_x));
            }
        }

        for (ih = it + 1; ih < GRAY_LEVELS; ih++)
        {
            /* Equation (4) in Ref. 1 */
            mu_x = 1.0 / (1.0 + term * std::abs(ih - mu_1[it]));
            if (!((mu_x < 1e-06) || (mu_x > 0.999999)))
            {
                /* Equation (6) & (8) in Ref. 1 */
                ent += data[ih] * (-mu_x * std::log(mu_x) - (1.0 - mu_x) * std::log(1.0 - mu_x));
            }
        }
        /* No need to divide by NUM_ROWS * NUM_COLS * LOG(2) ! */
        if (ent < min_ent)
        {
            min_ent = ent;
            threshold = it;
        }
    }
    return threshold;
}

static bool BimodalTest(std::vector<double>& y)
{
    int len = y.size();
    bool b = false;
    int modes = 0;

    for (int k = 1; k < len - 1; k++)
    {
        if (y[k - 1] < y[k] && y[k + 1] < y[k])
        {
            modes++;
            if (modes > 2)  return false;
        }
    }
    if (modes == 2) b = true;
    return b;
}

int Binarizate::Intermodes(const std::vector<int>& data)
{
    int maxbin = -1;
    for (int i = 0; i < data.size(); i++)
    {
        if (data[i] > 0) maxbin = i;
    }

    int minbin = -1;
    for (int i = data.size() - 1; i >= 0; i--)
    {
        if (data[i] > 0) minbin = i;
    }

    int length = (maxbin - minbin) + 1;
    std::vector<double> hist(data.size(), 0);
    for (int i = minbin; i <= maxbin; i++)
    {
        hist[i - minbin] = data[i];
    }

    int iter = 0;
    int threshold = -1;
    while (!BimodalTest(hist))
    {
        //smooth with a 3 point running mean filter
        double previous = 0;
        double current = 0;
        double next = hist[0];
        for (int i = 0; i < length - 1; i++)
        {
            previous = current;
            current = next;
            next = hist[i + 1];
            hist[i] = (previous + current + next) / 3;
        }
        hist[length - 1] = (current + next) / 3;
        iter++;
        if (iter > 10000)
        {
            threshold = -1;
            return threshold;
        }
    }

    // The threshold is the mean between the two peaks.
    int tt = 0;
    for (int i = 1; i < length - 1; i++)
    {
        if (hist[i - 1] < hist[i] && hist[i + 1] < hist[i])
        {
            tt += i;
        }
    }
    threshold = (int)std::floor(tt / 2.0);
    return threshold + minbin;
}

int Binarizate::IsoData(const std::vector<int>& data)
{
    int     i = 0;
    int     l = 0;
    int     totl = 0;
    int     g = 0;
    double  toth = 0.0;
    double  h = 0.0;
    const int GRAY_LEVELS = 256;
    for (i = 1; i < GRAY_LEVELS; i++)
    {
        if (data[i] > 0)
        {
            g = i + 1;
            break;
        }
    }
    while (true)
    {
        l = 0;
        totl = 0;
        for (i = 0; i < g; i++)
        {
            totl = totl + data[i];
            l = l + (data[i] * i);
        }
        h = 0;
        toth = 0;
        for (i = g + 1; i < GRAY_LEVELS; i++)
        {
            toth += data[i];
            h += ((double)data[i] * i);
        }
        if (totl > 0 && toth > 0)
        {
            l /= totl;
            h /= toth;
            if (g == (int)std::round((l + h) / 2.0))
                break;
        }
        g++;
        if (g > 254)    return -1;
    }
    return g;
}

int Binarizate::Li(const std::vector<int>& data)
{
    int threshold = 0;
    double num_pixels = 0;
    double sum_back = 0; /* sum of the background pixels at a given threshold */
    double sum_obj = 0;  /* sum of the object pixels at a given threshold */
    double num_back = 0; /* number of background pixels at a given threshold */
    double num_obj = 0;  /* number of object pixels at a given threshold */
    double old_thresh = 0;
    double new_thresh = 0;
    double mean_back = 0; /* mean of the background pixels at a given threshold */
    double mean_obj = 0;  /* mean of the object pixels at a given threshold */
    double mean = 0;  /* mean gray-level in the image */
    double tolerance = 0; /* threshold tolerance */
    double temp = 0;

    tolerance = 0.5;
    num_pixels = 0;
    for (int ih = 0; ih < 256; ih++)
        num_pixels += data[ih];

    /* Calculate the mean gray-level */
    mean = 0.0;
    for (int ih = 0 + 1; ih < 256; ih++) //0 + 1?
        mean += (double)ih * data[ih];
    mean /= num_pixels;
    /* Initial estimate */
    new_thresh = mean;

    do
    {
        old_thresh = new_thresh;
        threshold = (int)(old_thresh + 0.5);    /* range */
        /* Calculate the means of background and object pixels */
        /* Background */
        sum_back = 0;
        num_back = 0;
        for (int ih = 0; ih <= threshold; ih++)
        {
            sum_back += (double)ih * data[ih];
            num_back += data[ih];
        }
        mean_back = (num_back == 0 ? 0.0 : (sum_back / (double)num_back));
        /* Object */
        sum_obj = 0;
        num_obj = 0;
        for (int ih = threshold + 1; ih < 256; ih++)
        {
            sum_obj += (double)ih * data[ih];
            num_obj += data[ih];
        }
        mean_obj = (num_obj == 0 ? 0.0 : (sum_obj / (double)num_obj));

        /* Calculate the new threshold: Equation (7) in Ref. 2 */
        //new_thresh = simple_round ( ( mean_back - mean_obj ) / ( Math.log ( mean_back ) - Math.log ( mean_obj ) ) );
        //simple_round ( double x ) {
        // return ( int ) ( IS_NEG ( x ) ? x - .5 : x + .5 );
        //}
        //
        //#define IS_NEG( x ) ( ( x ) < -DBL_EPSILON ) 
        //DBL_EPSILON = 2.220446049250313E-16
        temp = (mean_back - mean_obj) / (std::log(mean_back) - std::log(mean_obj));

        if (temp < -2.220446049250313E-16)
            new_thresh = (int)(temp - 0.5);
        else
            new_thresh = (int)(temp + 0.5);
        /*  Stop the iterations when the difference between the
        new and old threshold values is less than the tolerance */
    } while (std::abs(new_thresh - old_thresh) > tolerance);
    return threshold;
}

int Binarizate::MaxEntropy(const std::vector<int>& data)
{
    const int GRAY_LEVELS = 256;
    int threshold = -1;
    int ih = 0;
    int it = 0;
    int first_bin = 0;
    int last_bin = 0;
    double tot_ent = 0.0;  /* total entropy */
    double max_ent = 0.0;  /* max entropy */
    double ent_back = 0.0; /* entropy of the background pixels at a given threshold */
    double ent_obj = 0.0;  /* entropy of the object pixels at a given threshold */

    std::vector<double> norm_histo(GRAY_LEVELS, 0.0);/* normalized histogram */
    std::vector<double> P1(GRAY_LEVELS, 0.0);/* cumulative normalized histogram */
    std::vector<double> P2(GRAY_LEVELS, 0.0);

    double total = 0;
    for (ih = 0; ih < GRAY_LEVELS; ih++)
        total += data[ih];

    for (ih = 0; ih < GRAY_LEVELS; ih++)
        norm_histo[ih] = data[ih] / total;

    P1[0] = norm_histo[0];
    P2[0] = 1.0 - P1[0];
    for (ih = 1; ih < GRAY_LEVELS; ih++)
    {
        P1[ih] = P1[ih - 1] + norm_histo[ih];
        P2[ih] = 1.0 - P1[ih];
    }

    /* Determine the first non-zero bin */
    first_bin = 0;
    for (ih = 0; ih < GRAY_LEVELS; ih++)
    {
        if (!(std::abs(P1[ih]) < 2.220446049250313E-16))
        {
            first_bin = ih;
            break;
        }
    }

    /* Determine the last non-zero bin */
    last_bin = 255;
    for (ih = 255; ih >= first_bin; ih--)
    {
        if (!(std::abs(P2[ih]) < 2.220446049250313E-16))
        {
            last_bin = ih;
            break;
        }
    }

    // Calculate the total entropy each gray-level
    // and find the threshold that maximizes it 
    max_ent = DBL_MIN;

    for (it = first_bin; it <= last_bin; it++)
    {
        /* Entropy of the background pixels */
        ent_back = 0.0;
        for (ih = 0; ih <= it; ih++)
        {
            if (data[ih] != 0)
            {
                ent_back -= (norm_histo[ih] / P1[it]) * std::log(norm_histo[ih] / P1[it]);
            }
        }

        /* Entropy of the object pixels */
        ent_obj = 0.0;
        for (ih = it + 1; ih < 256; ih++)
        {
            if (data[ih] != 0)
            {
                ent_obj -= (norm_histo[ih] / P2[it]) * std::log(norm_histo[ih] / P2[it]);
            }
        }

        /* Total entropy */
        tot_ent = ent_back + ent_obj;

        if (max_ent < tot_ent)
        {
            max_ent = tot_ent;
            threshold = it;
        }
    }
    return threshold;
}

int Binarizate::Mean(const std::vector<int>& data)
{
    const int GRAY_LEVELS = 256;
    int threshold = -1;
    double tot = 0, sum = 0;
    for (int i = 0; i < GRAY_LEVELS; i++)
    {
        tot += data[i];
        sum += ((double)i * data[i]);
    }
    threshold = (int)std::floor(sum / tot);
    return threshold;
}

/***************************************MinErrorI***************************************************/
static double A(const std::vector<int>& y, int j)
{
    if (j >= y.size()) j = y.size() - 1;
    double x = 0;
    for (int i = 0; i <= j; i++)
        x += y[i];
    return x;
}

static double B(const std::vector<int>& y, int j)
{
    if (j >= y.size()) j = y.size() - 1;
    double x = 0;
    for (int i = 0; i <= j; i++)
        x += i * y[i];
    return x;
}

static double C(const std::vector<int>& y, int j)
{
    if (j >= y.size()) j = y.size() - 1;
    double x = 0;
    for (int i = 0; i <= j; i++)
        x += i * i * y[i];
    return x;
}

int Binarizate::MinErrorI(const std::vector<int>& data)
{
    int threshold = Mean(data); //Initial estimate for the threshold is found with the MEAN algorithm.
    int Tprev = -2;
    double mu, nu, p, q, sigma2, tau2, w0, w1, w2, sqterm, temp;
    while (threshold != Tprev)
    {
        //Calculate some statistics.
        mu = B(data, threshold) / A(data, threshold);
        nu = (B(data, data.size() - 1) - B(data, threshold)) / (A(data, data.size() - 1) - A(data, threshold));
        p = A(data, threshold) / A(data, data.size() - 1);
        q = (A(data, data.size() - 1) - A(data, threshold)) / A(data, data.size() - 1);
        sigma2 = C(data, threshold) / A(data, threshold) - (mu * mu);
        tau2 = (C(data, data.size() - 1) - C(data, threshold)) / (A(data, data.size() - 1) - A(data, threshold)) - (nu * nu);

        //The terms of the quadratic equation to be solved.
        w0 = 1.0 / sigma2 - 1.0 / tau2;
        w1 = mu / sigma2 - nu / tau2;
        w2 = (mu * mu) / sigma2 - (nu * nu) / tau2 + std::log10((sigma2 * (q * q)) / (tau2 * (p * p)));

        //If the next threshold would be imaginary, return with the current one.
        sqterm = (w1 * w1) - w0 * w2;
        if (sqterm < 0)
        {
            return threshold;
        }

        //The updated threshold is the integer part of the solution of the quadratic equation.
        Tprev = threshold;
        temp = (w1 + std::sqrt(sqterm)) / w0;

        if (std::isnan(temp))
            threshold = Tprev;
        else
            threshold = (int)std::floor(temp);
    }
    return threshold;

}

int Binarizate::Minimum(const std::vector<int>& data)
{
    const int GRAY_LEVELS = 256;
    int iter = 0;
    int threshold = -1;
    std::vector<double> iHisto(GRAY_LEVELS, 0.0);
    for (int i = 0; i < GRAY_LEVELS; i++)
    {
        iHisto[i] = (double)data[i];
    }
    std::vector<double> tHisto(iHisto.size(), 0.0);

    while (!BimodalTest(iHisto))
    {
        //smooth with a 3 point running mean filter
        for (int i = 1; i < 255; i++)
        {
            tHisto[i] = (iHisto[i - 1] + iHisto[i] + iHisto[i + 1]) / 3;
        }
        tHisto[0] = (iHisto[0] + iHisto[1]) / 3;        //0 outside
        tHisto[255] = (iHisto[254] + iHisto[255]) / 3; //0 outside
        std::copy(tHisto.begin(), tHisto.end(), iHisto.begin());
        iter++;
        if (iter > 10000)
        {
            threshold = -1;
            return threshold;
        }
    }
    // The threshold is the minimum between the two peaks.
    for (int i = 1; i < 255; i++)
    {
        if (iHisto[i - 1] > iHisto[i] && iHisto[i + 1] >= iHisto[i])
        {
            threshold = i;
            break;
        }
    }
    return threshold;
}

int Binarizate::Moments(const std::vector<int>& data)
{
    const int GRAY_LEVELS = 256;
    double total = 0;
    double m0 = 1.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, sum = 0.0, p0 = 0.0;
    double cd, c0, c1, z0, z1;  /* auxiliary variables */
    int threshold = -1;

    //double[] histo = new  double[256];
    std::vector<double> histo(GRAY_LEVELS, 0.0);
    for (int i = 0; i < GRAY_LEVELS; i++)
    {
        total += data[i];
    }

    for (int i = 0; i < GRAY_LEVELS; i++)
    {
        histo[i] = (double)(data[i] / total); //normalised histogram
    }

    /* Calculate the first, second, and third order moments */
    for (int i = 0; i < GRAY_LEVELS; i++)
    {
        double di = i;
        m1 += di * histo[i];
        m2 += di * di * histo[i];
        m3 += di * di * di * histo[i];
    }
    /*
    First 4 moments of the gray-level image should match the first 4 moments
    of the target binary image. This leads to 4 equalities whose solutions
    are given in the Appendix of Ref. 1
    */
    cd = m0 * m2 - m1 * m1;
    c0 = (-m2 * m2 + m1 * m3) / cd;
    c1 = (m0 * -m3 + m2 * m1) / cd;
    z0 = 0.5 * (-c1 - std::sqrt(c1 * c1 - 4.0 * c0));
    z1 = 0.5 * (-c1 + std::sqrt(c1 * c1 - 4.0 * c0));
    p0 = (z1 - m1) / (z1 - z0);  /* Fraction of the object pixels in the target binary image */

    // The threshold is the gray-level closest  
    // to the p0-tile of the normalized histogram 
    sum = 0;
    for (int i = 0; i < GRAY_LEVELS; i++)
    {
        sum += histo[i];
        if (sum > p0)
        {
            threshold = i;
            break;
        }
    }
    return threshold;
}

int Binarizate::Otsu(const std::vector<int>& data)
{
    int k, kStar;  // k = the current threshold; kStar = optimal threshold
    double N1, N;    // N1 = # points with intensity <=k; N = total number of points
    double BCV, BCVmax; // The current Between Class Variance and maximum BCV
    double num, denom;  // temporary bookeeping
    double Sk;  // The total intensity for all histogram points <=k
    double S, L = 256; // The total intensity of the image

    // Initialize values:
    S = N = 0;
    for (k = 0; k < L; k++)
    {
        S += (double)k * data[k];   // Total histogram intensity
        N += data[k];       // Total number of data points
    }

    Sk = 0;
    N1 = data[0]; // The entry for zero intensity
    BCV = 0;
    BCVmax = 0;
    kStar = 0;

    // Look at each possible threshold value,
    // calculate the between-class variance, and decide if it's a max
    for (k = 1; k < L - 1; k++)
    {
        // No need to check endpoints k = 0 or k = L-1
        Sk += (double)k * data[k];
        N1 += data[k];

        // The float casting here is to avoid compiler warning about loss of precision and
        // will prevent overflow in the case of large saturated images
        denom = (double)(N1) * (N - N1); // Maximum value of denom is (N^2)/4 =  approx. 3E10

        if (denom != 0)
        {
            // Float here is to avoid loss of precision when dividing
            num = ((double)N1 / N) * S - Sk;    // Maximum value of num =  255*N = approx 8E7
            BCV = (num * num) / denom;
        }
        else
        {
            BCV = 0;
        }

        if (BCV >= BCVmax)
        { // Assign the best threshold found so far
            BCVmax = BCV;
            kStar = k;
        }
    }
    // kStar += 1;  // Use QTI convention that intensity -> 1 if intensity >= k
    // (the algorithm was developed for I-> 1 if I <= k.)
    return kStar;
}

static double PartialSum(const std::vector<int>& y, int j)
{
    double x = 0;
    for (int i = 0; i <= j; i++)
        x += y[i];
    return x;
}

int Binarizate::Percentile(const std::vector<int>& data)
{
    int iter = 0;
    int threshold = -1;
    double ptile = 0.5; // default fraction of foreground pixels
    std::vector<double> avec(256, 0.0);
    for (int i = 0; i < 256; i++)
    {
        avec[i] = 0.0;
    }

    double total = PartialSum(data, 255);
    double temp = 1.0;
    for (int i = 0; i < 256; i++)
    {
        avec[i] = std::abs((PartialSum(data, i) / total) - ptile);
        if (avec[i] < temp)
        {
            temp = avec[i];
            threshold = i;
        }
    }
    return threshold;
}

int Binarizate::RenyiEntropy(const std::vector<int>& data)
{
    int threshold;
    int opt_threshold;

    int ih, it;
    int first_bin;
    int last_bin;
    int tmp_var;
    int t_star1, t_star2, t_star3;
    int beta1, beta2, beta3;
    double alpha;/* alpha parameter of the method */
    double term;
    double tot_ent;  /* total entropy */
    double max_ent;  /* max entropy */
    double ent_back; /* entropy of the background pixels at a given threshold */
    double ent_obj;  /* entropy of the object pixels at a given threshold */
    double omega;
    std::vector<double> norm_histo(256, 0.0);/* normalized histogram */
    std::vector<double> P1(256, 0.0);/* cumulative normalized histogram */
    std::vector<double> P2(256, 0.0);

    double total = 0;
    for (ih = 0; ih < 256; ih++)
        total += data[ih];

    for (ih = 0; ih < 256; ih++)
        norm_histo[ih] = data[ih] / total;

    P1[0] = norm_histo[0];
    P2[0] = 1.0 - P1[0];
    for (ih = 1; ih < 256; ih++)
    {
        P1[ih] = P1[ih - 1] + norm_histo[ih];
        P2[ih] = 1.0 - P1[ih];
    }

    /* Determine the first non-zero bin */
    first_bin = 0;
    for (ih = 0; ih < 256; ih++)
    {
        if (!(std::abs(P1[ih]) < 2.220446049250313E-16))
        {
            first_bin = ih;
            break;
        }
    }

    /* Determine the last non-zero bin */
    last_bin = 255;
    for (ih = 255; ih >= first_bin; ih--)
    {
        if (!(std::abs(P2[ih]) < 2.220446049250313E-16))
        {
            last_bin = ih;
            break;
        }
    }

    /* Maximum Entropy Thresholding - BEGIN */
    /* ALPHA = 1.0 */
    /* Calculate the total entropy each gray-level
    and find the threshold that maximizes it
    */
    threshold = 0; // was MIN_INT in original code, but if an empty image is processed it gives an error later on.
    max_ent = 0.0;

    for (it = first_bin; it <= last_bin; it++)
    {
        /* Entropy of the background pixels */
        ent_back = 0.0;
        for (ih = 0; ih <= it; ih++)
        {
            if (data[ih] != 0) {
                ent_back -= (norm_histo[ih] / P1[it]) * std::log(norm_histo[ih] / P1[it]);
            }
        }

        /* Entropy of the object pixels */
        ent_obj = 0.0;
        for (ih = it + 1; ih < 256; ih++)
        {
            if (data[ih] != 0) {
                ent_obj -= (norm_histo[ih] / P2[it]) * std::log(norm_histo[ih] / P2[it]);
            }
        }

        /* Total entropy */
        tot_ent = ent_back + ent_obj;

        if (max_ent < tot_ent)
        {
            max_ent = tot_ent;
            threshold = it;
        }
    }
    t_star2 = threshold;

    /* Maximum Entropy Thresholding - END */
    threshold = 0; //was MIN_INT in original code, but if an empty image is processed it gives an error later on.
    max_ent = 0.0;
    alpha = 0.5;
    term = 1.0 / (1.0 - alpha);
    for (it = first_bin; it <= last_bin; it++)
    {
        /* Entropy of the background pixels */
        ent_back = 0.0;
        for (ih = 0; ih <= it; ih++)
            ent_back += std::sqrt(norm_histo[ih] / P1[it]);

        /* Entropy of the object pixels */
        ent_obj = 0.0;
        for (ih = it + 1; ih < 256; ih++)
            ent_obj += std::sqrt(norm_histo[ih] / P2[it]);

        /* Total entropy */
        tot_ent = term * ((ent_back * ent_obj) > 0.0 ? std::log(ent_back * ent_obj) : 0.0);

        if (tot_ent > max_ent)
        {
            max_ent = tot_ent;
            threshold = it;
        }
    }

    t_star1 = threshold;

    threshold = 0; //was MIN_INT in original code, but if an empty image is processed it gives an error later on.
    max_ent = 0.0;
    alpha = 2.0;
    term = 1.0 / (1.0 - alpha);
    for (it = first_bin; it <= last_bin; it++)
    {
        /* Entropy of the background pixels */
        ent_back = 0.0;
        for (ih = 0; ih <= it; ih++)
            ent_back += (norm_histo[ih] * norm_histo[ih]) / (P1[it] * P1[it]);

        /* Entropy of the object pixels */
        ent_obj = 0.0;
        for (ih = it + 1; ih < 256; ih++)
            ent_obj += (norm_histo[ih] * norm_histo[ih]) / (P2[it] * P2[it]);

        /* Total entropy */
        tot_ent = term * ((ent_back * ent_obj) > 0.0 ? std::log(ent_back * ent_obj) : 0.0);

        if (tot_ent > max_ent)
        {
            max_ent = tot_ent;
            threshold = it;
        }
    }

    t_star3 = threshold;

    /* Sort t_star values */
    if (t_star2 < t_star1)
    {
        tmp_var = t_star1;
        t_star1 = t_star2;
        t_star2 = tmp_var;
    }
    if (t_star3 < t_star2)
    {
        tmp_var = t_star2;
        t_star2 = t_star3;
        t_star3 = tmp_var;
    }
    if (t_star2 < t_star1)
    {
        tmp_var = t_star1;
        t_star1 = t_star2;
        t_star2 = tmp_var;
    }

    /* Adjust beta values */
    if (std::abs(t_star1 - t_star2) <= 5)
    {
        if (std::abs(t_star2 - t_star3) <= 5)
        {
            beta1 = 1;
            beta2 = 2;
            beta3 = 1;
        }
        else
        {
            beta1 = 0;
            beta2 = 1;
            beta3 = 3;
        }
    }
    else
    {
        if (std::abs(t_star2 - t_star3) <= 5)
        {
            beta1 = 3;
            beta2 = 1;
            beta3 = 0;
        }
        else
        {
            beta1 = 1;
            beta2 = 2;
            beta3 = 1;
        }
    }
    /* Determine the optimal threshold value */
    omega = P1[t_star3] - P1[t_star1];
    opt_threshold = (int)(t_star1 * (P1[t_star1] + 0.25 * omega * beta1) + 0.25 * t_star2 * omega * beta2 + t_star3 * (P2[t_star3] + 0.25 * omega * beta3));

    return opt_threshold;

}

int Binarizate::Shanbhag(const std::vector<int>& data)
{
    int threshold;
    int ih, it;
    int first_bin;
    int last_bin;
    double term;
    double tot_ent;  /* total entropy */
    double min_ent;  /* max entropy */
    double ent_back; /* entropy of the background pixels at a given threshold */
    double ent_obj;  /* entropy of the object pixels at a given threshold */
    std::vector<double> norm_histo(256, 0.0);/* normalized histogram */
    std::vector<double> P1(256, 0.0);/* cumulative normalized histogram */
    std::vector<double> P2(256, 0.0);

    double total = 0;
    for (ih = 0; ih < 256; ih++)
        total += data[ih];

    for (ih = 0; ih < 256; ih++)
        norm_histo[ih] = data[ih] / total;

    P1[0] = norm_histo[0];
    P2[0] = 1.0 - P1[0];
    for (ih = 1; ih < 256; ih++) {
        P1[ih] = P1[ih - 1] + norm_histo[ih];
        P2[ih] = 1.0 - P1[ih];
    }

    /* Determine the first non-zero bin */
    first_bin = 0;
    for (ih = 0; ih < 256; ih++)
    {
        if (!(std::abs(P1[ih]) < 2.220446049250313E-16))
        {
            first_bin = ih;
            break;
        }
    }

    /* Determine the last non-zero bin */
    last_bin = 255;
    for (ih = 255; ih >= first_bin; ih--)
    {
        if (!(std::abs(P2[ih]) < 2.220446049250313E-16))
        {
            last_bin = ih;
            break;
        }
    }

    // Calculate the total entropy each gray-level
    // and find the threshold that maximizes it 
    threshold = -1;
    min_ent = DBL_MAX;

    for (it = first_bin; it <= last_bin; it++)
    {
        /* Entropy of the background pixels */
        ent_back = 0.0;
        term = 0.5 / P1[it];
        for (ih = 1; ih <= it; ih++)
        { //0+1?
            ent_back -= norm_histo[ih] * std::log(1.0 - term * P1[ih - 1]);
        }
        ent_back *= term;

        /* Entropy of the object pixels */
        ent_obj = 0.0;
        term = 0.5 / P2[it];
        for (ih = it + 1; ih < 256; ih++)
        {
            ent_obj -= norm_histo[ih] * std::log(1.0 - term * P2[ih]);
        }
        ent_obj *= term;

        /* Total entropy */
        tot_ent = std::abs(ent_back - ent_obj);

        if (tot_ent < min_ent)
        {
            min_ent = tot_ent;
            threshold = it;
        }
    }
    return threshold;
}

int Binarizate::Triangle(std::vector<int>& data)
{
    const int GRAY_LEVELS = 256;

    // find min and max
    int min = 0, dmax = 0, max = 0, min2 = 0;
    for (int i = 0; i < data.size(); i++)
    {
        if (data[i] > 0)
        {
            min = i;
            break;
        }
    }
    if (min > 0) min--; // line to the (p==0) point, not to data[min]

    // The Triangle algorithm cannot tell whether the data is skewed to one side or another.
    // This causes a problem as there are 2 possible thresholds between the max and the 2 extremes
    // of the histogram.
    // Here I propose to find out to which side of the max point the data is furthest, and use that as
    //  the other extreme.
    for (int i = 255; i > 0; i--)
    {
        if (data[i] > 0)
        {
            min2 = i;
            break;
        }
    }
    if (min2 < 255) min2++; // line to the (p==0) point, not to data[min]

    for (int i = 0; i < GRAY_LEVELS; i++)
    {
        if (data[i] > dmax)
        {
            max = i;
            dmax = data[i];
        }
    }
    // find which is the furthest side
    //IJ.log(""+min+" "+max+" "+min2);
    bool inverted = false;
    if ((max - min) < (min2 - max))
    {
        // reverse the histogram
        inverted = true;
        int left = 0;          // index of leftmost element
        int right = 255;        // index of rightmost element
        while (left < right)
        {
            // exchange the left and right elements
            int temp = data[left];
            data[left] = data[right];
            data[right] = temp;
            // move the bounds toward the center
            left++;
            right--;
        }
        min = 255 - min2;
        max = 255 - max;
    }

    if (min == max)
    {
        return min;
    }

    // describe line by nx * x + ny * y - d = 0
    double nx, ny, d;
    // nx is just the max frequency as the other point has freq=0
    nx = data[max];   //-min; // data[min]; //  lowest value bmin = (p=0)% in the image
    ny = min - max;
    d = std::sqrt(nx * nx + ny * ny);
    nx /= d;
    ny /= d;
    d = nx * min + ny * data[min];

    // find split point
    int split = min;
    double splitDistance = 0;
    for (int i = min + 1; i <= max; i++)
    {
        double newDistance = nx * i + ny * data[i] - d;
        if (newDistance > splitDistance)
        {
            split = i;
            splitDistance = newDistance;
        }
    }
    split--;

    if (inverted)
    {
        // The histogram might be used for something else, so let's reverse it back
        int left = 0;
        int right = 255;
        while (left < right)
        {
            int temp = data[left];
            data[left] = data[right];
            data[right] = temp;
            left++;
            right--;
        }
        return (255 - split);
    }
    else
        return split;

}

int Binarizate::Yen(const std::vector<int>& data)
{
    const int GRAY_LEVELS = 256;
    int threshold = 0;
    int ih = 0;
    int it = 0;
    double crit = 0.0;
    double max_crit = 0.0;
    std::vector<double> norm_histo(GRAY_LEVELS, 0.0);/* normalized histogram */
    std::vector<double> P1(GRAY_LEVELS, 0.0);
    std::vector<double> P1_sq(GRAY_LEVELS, 0.0);
    std::vector<double> P2_sq(GRAY_LEVELS, 0.0);

    double total = 0;
    for (ih = 0; ih < GRAY_LEVELS; ih++)
    {
        total += data[ih];
    }

    for (ih = 0; ih < GRAY_LEVELS; ih++)
    {
        norm_histo[ih] = data[ih] / total;
    }

    P1[0] = norm_histo[0];
    for (ih = 1; ih < GRAY_LEVELS; ih++)
    {
        P1[ih] = P1[ih - 1] + norm_histo[ih];
    }

    P1_sq[0] = norm_histo[0] * norm_histo[0];
    for (ih = 1; ih < GRAY_LEVELS; ih++)
    {
        P1_sq[ih] = P1_sq[ih - 1] + norm_histo[ih] * norm_histo[ih];
    }

    P2_sq[255] = 0.0;
    for (ih = 254; ih >= 0; ih--)
    {
        P2_sq[ih] = P2_sq[ih + 1] + norm_histo[ih + 1] * norm_histo[ih + 1];
    }

    /* Find the threshold that maximizes the criterion */
    threshold = -1;
    max_crit = DBL_MIN;
    for (it = 0; it < GRAY_LEVELS; it++)
    {
        crit = -1.0 * ((P1_sq[it] * P2_sq[it]) > 0.0 ? std::log(P1_sq[it] * P2_sq[it]) : 0.0) + 2 * ((P1[it] * (1.0 - P1[it])) > 0.0 ? std::log(P1[it] * (1.0 - P1[it])) : 0.0);
        if (crit > max_crit)
        {
            max_crit = crit;
            threshold = it;
        }
    }
    return threshold;
}
