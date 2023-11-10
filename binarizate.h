#pragma once
#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>


using namespace cv;
#define PROCESS_IMG_SUCESS 0
#define PROCESS_IMG_FAIL 1


	//二值化
	class Binarizate
	{


	public:
		static int OTSU(cv::Mat srcImage);// 大律法函数实现
		static int OtsuBinarizate(cv::Mat srcImage, cv::Mat& dstImage, int maxVal, int threshType);//大律法二值化
		static int FixedBinarizate(cv::Mat srcImage, cv::Mat& dstImage, int thresh, int maxVal, int threshType);//固定化阈值
		static int AdaptiveBinarizate(cv::Mat srcImage, cv::Mat& dstImage, int maxVal, int adaptiveMethod, int thresholdType, int blockSize, int constValue);//自适应阈值化


    public:
        /** @brief 统计每个灰度级及其对应的像素个数

        @param src CV_8UC1类型的图像
        @param data 所得到的结果, 索引为灰度级, 对应值为像素个数. 内有256个元素
        */
        static void CalcHist(const cv::Mat& src, std::vector<int>& data);


    public:
        static int DefaultIsoData(const std::vector<int>& data);

        /** @brief original ImageJ IsoData implementation

        函数有可能会修改data
        */
        static int IJIsoData(std::vector<int>& data);

        /** @brief Implements Huang's fuzzy thresholding method

        Uses Shannon's entropy function (one can also use Yager's entropy function)
        Huang L.-K. and Wang M.-J.J. (1995) "Image Thresholding by Minimizing
        the Measures of Fuzziness" Pattern Recognition, 28(1): 41-51
        M. Emre Celebi  06.15.2007
        Ported to ImageJ plugin by G. Landini from E Celebi's fourier_0.8 routines
        */
        static int Huang(const std::vector<int>& data);

        /**  @brief Implements Intermodes thresholding method

        J. M. S. Prewitt and M. L. Mendelsohn, "The analysis of cell images," in
        Annals of the New York Academy of Sciences, vol. 128, pp. 1035-1053, 1966.
        ported to ImageJ plugin by G.Landini from Antti Niemisto's Matlab code (GPL)
        Original Matlab code Copyright (C) 2004 Antti Niemisto
        See http://www.cs.tut.fi/~ant/histthresh/ for an excellent slide presentation
        and the original Matlab code.

        Assumes a bimodal histogram. The histogram needs is smoothed (using a
        running average of size 3, iteratively) until there are only two local maxima.
        j and k
        Threshold t is (j+k)/2.
        Images with histograms having extremely unequal peaks or a broad and
        flat valleys are unsuitable for this method.
        */
        static int Intermodes(const std::vector<int>& data);

        /** @brief Implements IsoData thresholding method
        Also called intermeans
        Iterative procedure based on the isodata algorithm [T.W. Ridler, S. Calvard, Picture
        thresholding using an iterative selection method, IEEE Trans. System, Man and
        Cybernetics, SMC-8 (1978) 630-632.]
        The procedure divides the image into objects and background by taking an initial threshold,
        then the averages of the pixels at or below the threshold and pixels above are computed.
        The averages of those two values are computed, the threshold is incremented and the
        process is repeated until the threshold is larger than the composite average. That is,
         threshold = (average background + average objects)/2
        The code in ImageJ that implements this function is the getAutoThreshold() method in the ImageProcessor class.
        From: Tim Morris (dtm@ap.co.umist.ac.uk)
        Subject: Re: Thresholding method?
        posted to sci.image.processing on 1996/06/24
        The algorithm implemented in NIH Image sets the threshold as that grey
        value, G, for which the average of the averages of the grey values
        below and above G is equal to G. It does this by initialising G to the
        lowest sensible value and iteratingL = the average grey value of pixels with intensities < G
        H = the average grey value of pixels with intensities > G
        is G = (L + H)/2?
        yes => exit
        no => increment G and repeat
        */
        static int IsoData(const std::vector<int>& data);

        /** @brief Implements Li's Minimum Cross Entropy thresholding method

        This implementation is based on the iterative version (Ref. 2) of the algorithm.
        1) Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
            Pattern Recognition, 26(4): 617-625
        2) Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
            Cross Entropy Thresholding"Pattern Recognition Letters, 18(8): 771-776
        3) Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
            Techniques and Quantitative Performance Evaluation" Journal of
            Electronic Imaging, 13(1): 146-165
            http://citeseer.ist.psu.edu/sezgin04survey.html
        Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
        */
        static int Li(const std::vector<int>& data);

        /** @brief Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
        Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for
        Gray-Level Picture Thresholding Using the Entropy of the Histogram"
        Graphical Models and Image Processing, 29(3): 273-285
        M. Emre Celebi
        06.15.2007
        Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
        */
        static int MaxEntropy(const std::vector<int>& data);

        /** @brief Implements Mean thresholding method

        C. A. Glasbey, "An analysis of histogram-based thresholding algorithms,"
        CVGIP: Graphical Models and Image Processing, vol. 55, pp. 532-537, 1993.
        The threshold is the mean of the greyscale data
        */
        static int Mean(const std::vector<int>& data);

        /** @brief Implements Minimum error thresholding thresholding method

        Kittler and J. Illingworth, "Minimum error thresholding," Pattern Recognition, vol. 19, pp. 41-47, 1986.
        C. A. Glasbey, "An analysis of histogram-based thresholding algorithms," CVGIP: Graphical Models and Image Processing, vol. 55, pp. 532-537, 1993.
        Ported to ImageJ plugin by G.Landini from Antti Niemisto's Matlab code (GPL)
        Original Matlab code Copyright (C) 2004 Antti Niemisto
        See http://www.cs.tut.fi/~ant/histthresh/ for an excellent slide presentation
        and the original Matlab code.
        */
        static int MinErrorI(const std::vector<int>& data);

        /**
        J. M. S. Prewitt and M. L. Mendelsohn, "The analysis of cell images," in
        Annals of the New York Academy of Sciences, vol. 128, pp. 1035-1053, 1966.
        ported to ImageJ plugin by G.Landini from Antti Niemisto's Matlab code (GPL)
        Original Matlab code Copyright (C) 2004 Antti Niemisto
        See http://www.cs.tut.fi/~ant/histthresh/ for an excellent slide presentation
        and the original Matlab code.
        Assumes a bimodal histogram. The histogram needs is smoothed (using a
        running average of size 3, iteratively) until there are only two local maxima.
        Threshold t is such that yt-1 > yt <= yt+1.
        Images with histograms having extremely unequal peaks or a broad and
        flat valleys are unsuitable for this method.
        */
        static int Minimum(const std::vector<int>& data);

        /**

        W. Tsai, "Moment-preserving thresholding: a new approach," Computer Vision,
        Graphics, and Image Processing, vol. 29, pp. 377-393, 1985.
        Ported to ImageJ plugin by G.Landini from the the open source project FOURIER 0.8
        by  M. Emre Celebi , Department of Computer Science,  Louisiana State University in Shreveport
        Shreveport, LA 71115, USA
        http://sourceforge.net/projects/fourier-ipal
        http://www.lsus.edu/faculty/~ecelebi/fourier.htm
        */
        static int Moments(const std::vector<int>& data);

        /**
        Otsu's threshold algorithm
        C++ code by Jordan Bevik <Jordan.Bevic@qtiworld.com>
        ported to ImageJ plugin by G.Landini
        */
        static int Otsu(const std::vector<int>& data);

        /**
        W. Doyle, "Operation useful for similarity-invariant pattern recognition,"
        Journal of the Association for Computing Machinery, vol. 9,pp. 259-267, 1962.
        ported to ImageJ plugin by G.Landini from Antti Niemisto's Matlab code (GPL)
        Original Matlab code Copyright (C) 2004 Antti Niemisto
        See http://www.cs.tut.fi/~ant/histthresh/ for an excellent slide presentation
        and the original Matlab code.
        */
        static int Percentile(const std::vector<int>& data);

        /**
        Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for
        Gray-Level Picture Thresholding Using the Entropy of the Histogram"
        Graphical Models and Image Processing, 29(3): 273-285
        M. Emre Celebi
        06.15.2007
        Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
        */
        static int RenyiEntropy(const std::vector<int>& data);

        /**

        Shanhbag A.G. (1994) "Utilization of Information Measure as a Means of
        Image Thresholding" Graphical Models and Image Processing, 56(5): 414-419
        Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
        */
        static int Shanbhag(const std::vector<int>& data);

        /**
        Zack, G. W., Rogers, W. E. and Latt, S. A., 1977,
        Automatic Measurement of Sister Chromatid Exchange Frequency,
        Journal of Histochemistry and Cytochemistry 25 (7), pp. 741-753

        modified from Johannes Schindelin plugin
        函数有可能会修改data
        */
        static int Triangle(std::vector<int>& data);

        /** @brief Implements Yen  thresholding method

         1) Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion for Automatic Multilevel Thresholding"
            IEEE Trans. on Image Processing, 4(3): 370-378
         2) Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
            Techniques and Quantitative Performance Evaluation" Journal of
            Electronic Imaging, 13(1): 146-165
            http://citeseer.ist.psu0.edu/sezgin04survey.html

        M. Emre Celebi
        06.15.2007
        Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routin
        */
        static int Yen(const std::vector<int>& data);

    };
