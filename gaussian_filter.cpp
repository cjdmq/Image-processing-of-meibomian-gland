#include"gaussian_fliter.h"


cv::Mat gaussianfilter::image_make_border(cv::Mat& src)
{
    int w = cv::getOptimalDFTSize(src.cols);
    int h = cv::getOptimalDFTSize(src.rows);
    cv::Mat padded;
    copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    padded.convertTo(padded, CV_32FC1);
    return padded;
}

//Ƶ�����˲�
cv::Mat gaussianfilter::frequency_filter(cv::Mat& scr, cv::Mat& blur)
{
    //***********************DFT*******************
    cv::Mat plane[] = { scr, cv::Mat::zeros(scr.size() , CV_32FC1) }; //����ͨ�����洢dft���ʵ�����鲿��CV_32F������Ϊ��ͨ������
    cv::Mat complexIm;
    merge(plane, 2, complexIm);//�ϲ�ͨ�� ������������ϲ�Ϊһ��2ͨ����Mat��������
    dft(complexIm, complexIm);//���и���Ҷ�任���������������

    //***************���Ļ�********************
    split(complexIm, plane);//����ͨ����������룩
    //    plane[0] = plane[0](Rect(0, 0, plane[0].cols & -2, plane[0].rows & -2));//����Ϊʲô&��-2����鿴opencv�ĵ�
    //    //��ʵ��Ϊ�˰��к��б��ż�� -2�Ķ�������11111111.......10 ���һλ��0
    int cx = plane[0].cols / 2; int cy = plane[0].rows / 2;//���µĲ������ƶ�ͼ��  (��Ƶ�Ƶ�����)
    cv::Mat part1_r(plane[0], cv::Rect(0, 0, cx, cy));  //Ԫ�������ʾΪ(cx,cy)
    cv::Mat part2_r(plane[0], cv::Rect(cx, 0, cx, cy));
    cv::Mat part3_r(plane[0], cv::Rect(0, cy, cx, cy));
    cv::Mat part4_r(plane[0], cv::Rect(cx, cy, cx, cy));

    cv::Mat temp;
    part1_r.copyTo(temp);  //���������½���λ��(ʵ��)
    part4_r.copyTo(part1_r);
    temp.copyTo(part4_r);

    part2_r.copyTo(temp);  //���������½���λ��(ʵ��)
    part3_r.copyTo(part2_r);
    temp.copyTo(part3_r);

    cv::Mat part1_i(plane[1], cv::Rect(0, 0, cx, cy));  //Ԫ������(cx,cy)
    cv::Mat part2_i(plane[1], cv::Rect(cx, 0, cx, cy));
    cv::Mat part3_i(plane[1], cv::Rect(0, cy, cx, cy));
    cv::Mat part4_i(plane[1], cv::Rect(cx, cy, cx, cy));

    part1_i.copyTo(temp);  //���������½���λ��(�鲿)
    part4_i.copyTo(part1_i);
    temp.copyTo(part4_i);

    part2_i.copyTo(temp);  //���������½���λ��(�鲿)
    part3_i.copyTo(part2_i);
    temp.copyTo(part3_i);

    //*****************�˲���������DFT����ĳ˻�****************
    cv::Mat blur_r, blur_i, BLUR;
    multiply(plane[0], blur, blur_r); //�˲���ʵ�����˲���ģ���ӦԪ����ˣ�
    multiply(plane[1], blur, blur_i);//�˲����鲿���˲���ģ���ӦԪ����ˣ�
    cv::Mat plane1[] = { blur_r, blur_i };
    merge(plane1, 2, BLUR);//ʵ�����鲿�ϲ�

    //*********************�õ�ԭͼƵ��ͼ***********************************
    magnitude(plane[0], plane[1], plane[0]);//��ȡ����ͼ��0ͨ��Ϊʵ��ͨ����1Ϊ�鲿����Ϊ��ά����Ҷ�任����Ǹ���
    plane[0] += cv::Scalar::all(1);  //����Ҷ�任���ͼƬ���÷��������ж�����������ȽϺÿ�
    log(plane[0], plane[0]);    // float�͵ĻҶȿռ�Ϊ[0��1])
    normalize(plane[0], plane[0], 1, 0, CV_MINMAX);  //��һ��������ʾ

    idft(BLUR, BLUR);    //idft���ҲΪ����
    split(BLUR, plane);//����ͨ������Ҫ��ȡͨ��
    magnitude(plane[0], plane[1], plane[0]);  //���ֵ(ģ)
    normalize(plane[0], plane[0], 1, 0, CV_MINMAX);  //��һ��������ʾ
    return plane[0];//���ز���
}

//*****************�����ͨ�˲���***********************
cv::Mat gaussianfilter::ideal_low_kernel(cv::Mat& scr, float sigma)
{
    cv::Mat ideal_low_pass(scr.size(), CV_32FC1); //��CV_32FC1
    float d0 = sigma;//�뾶D0ԽС��ģ��Խ�󣻰뾶D0Խ��ģ��ԽС
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            double d = sqrt(pow((i - scr.rows / 2), 2) + pow((j - scr.cols / 2), 2));//����,����pow����Ϊfloat��
            if (d <= d0) {
                ideal_low_pass.at<float>(i, j) = 1;
            }
            else {
                ideal_low_pass.at<float>(i, j) = 0;
            }
        }
    }
    std::string name = "�����ͨ�˲���d0=" + std::to_string(sigma);
    /*cv::namedWindow(name, cv::WINDOW_FREERATIO);
    imshow(name, ideal_low_pass);*/
    return ideal_low_pass;
}

//�����ͨ�˲���
cv::Mat gaussianfilter::ideal_low_pass_filter(cv::Mat& src, float sigma)
{
    cv::Mat padded = image_make_border(src);
    cv::Mat ideal_kernel = ideal_low_kernel(padded, sigma);
    cv::Mat result = frequency_filter(padded, ideal_kernel);
    return result;
}


cv::Mat gaussianfilter::butterworth_low_kernel(cv::Mat& scr, float sigma, int n)
{
    cv::Mat butterworth_low_pass(scr.size(), CV_32FC1); //��CV_32FC1
    double D0 = sigma;//�뾶D0ԽС��ģ��Խ�󣻰뾶D0Խ��ģ��ԽС
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            double d = sqrt(pow((i - scr.rows / 2), 2) + pow((j - scr.cols / 2), 2));//����,����pow����Ϊfloat��
            butterworth_low_pass.at<float>(i, j) = 1.0 / (1 + pow(d / D0, 2 * n));
        }
    }

    std::string name = "������˹��ͨ�˲���d0=" + std::to_string(sigma) + "n=" + std::to_string(n);
    /*cv::namedWindow(name, cv::WINDOW_FREERATIO);
    imshow(name, butterworth_low_pass);*/
    return butterworth_low_pass;
}

//������˹��ͨ�˲���
cv::Mat gaussianfilter::butterworth_low_paass_filter(cv::Mat& src, float d0, int n)
{
    //H = 1 / (1+(D/D0)^2n)    n��ʾ������˹�˲����Ĵ���
    //����n=1 ������͸�ֵ    ����n=2 ��΢����͸�ֵ  ����n=5 ��������͸�ֵ   ����n=20 ��ILPF����
    cv::Mat padded = image_make_border(src);
    cv::Mat butterworth_kernel = butterworth_low_kernel(padded, d0, n);
    cv::Mat result = frequency_filter(padded, butterworth_kernel);
    return result;
}

cv::Mat gaussianfilter::gaussian_low_pass_kernel(cv::Mat scr, float sigma)
{
    cv::Mat gaussianBlur(scr.size(), CV_32FC1); //��CV_32FC1
    float d0 = 2 * sigma * sigma;//��˹����������ԽС��Ƶ�ʸ�˹�˲���Խխ���˳���Ƶ�ɷ�Խ�࣬ͼ���Խƽ��
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            float d = pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2);//����,����pow����Ϊfloat��
            gaussianBlur.at<float>(i, j) = expf(-d / d0);//expfΪ��eΪ�����ݣ�����Ϊfloat�ͣ�
        }
    }
    //    Mat show = gaussianBlur.clone();
    //    //��һ����[0,255]����ʾ
    //    normalize(show, show, 0, 255, NORM_MINMAX);
    //    //ת����CV_8U��
    //    show.convertTo(show, CV_8U);
    //    std::string pic_name = "gaussi" + std::to_string((int)sigma) + ".jpg";
    //    imwrite( pic_name, show);
    /*cv::namedWindow("��˹��ͨ�˲���", cv::WINDOW_FREERATIO);
    imshow("��˹��ͨ�˲���", gaussianBlur);*/
    return gaussianBlur;
}

//��˹��ͨ
cv::Mat gaussianfilter::gaussian_low_pass_filter(cv::Mat& src, float d0)
{
    cv::Mat padded = image_make_border(src);
    cv::Mat gaussian_kernel = gaussian_low_pass_kernel(padded, d0);//�����ͨ�˲���
    cv::Mat result = frequency_filter(padded, gaussian_kernel);
    return result;
}

cv::Mat gaussianfilter::ideal_high_kernel(cv::Mat& scr, float sigma)
{
    cv::Mat ideal_high_pass(scr.size(), CV_32FC1); //��CV_32FC1
    float d0 = sigma;//�뾶D0ԽС��ģ��Խ�󣻰뾶D0Խ��ģ��ԽС
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            double d = sqrt(pow((i - scr.rows / 2), 2) + pow((j - scr.cols / 2), 2));//����,����pow����Ϊfloat��
            if (d <= d0) {
                ideal_high_pass.at<float>(i, j) = 0;
            }
            else {
                ideal_high_pass.at<float>(i, j) = 1;
            }
        }
    }
    std::string name = "�����ͨ�˲���d0=" + std::to_string(sigma);
    /*cv::namedWindow(name, cv::WINDOW_FREERATIO);
    imshow(name, ideal_high_pass);*/
    return ideal_high_pass;
}

//�����ͨ�˲���
cv::Mat gaussianfilter::ideal_high_pass_filter(cv::Mat& src, float sigma)
{
    cv::Mat padded = image_make_border(src);
    cv::Mat ideal_kernel = ideal_high_kernel(padded, sigma);
    cv::Mat result = frequency_filter(padded, ideal_kernel);
    return result;
}

cv::Mat gaussianfilter::butterworth_high_kernel(cv::Mat& scr, float sigma, int n)
{
    cv::Mat butterworth_low_pass(scr.size(), CV_32FC1); //��CV_32FC1
    double D0 = sigma;//�뾶D0ԽС��ģ��Խ�󣻰뾶D0Խ��ģ��ԽС
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            double d = sqrt(pow((i - scr.rows / 2), 2) + pow((j - scr.cols / 2), 2));//����,����pow����Ϊfloat��
            butterworth_low_pass.at<float>(i, j) = 1.0 / (1 + pow(D0 / d, 2 * n));
        }
    }

    std::string name = "������˹��ͨ�˲���d0=" + std::to_string(sigma) + "n=" + std::to_string(n);
    /*cv::namedWindow(name, cv::WINDOW_FREERATIO);
    imshow(name, butterworth_low_pass);*/
    return butterworth_low_pass;
}

//������˹��ͨ�˲���
cv::Mat gaussianfilter::butterworth_high_paass_filter(cv::Mat& src, float d0, int n)
{
    //H = 1 / (1+(D0/D)^2n)    n��ʾ������˹�˲����Ĵ���
    cv::Mat padded = image_make_border(src);
    cv::Mat butterworth_kernel = butterworth_high_kernel(padded, d0, n);
    cv::Mat result = frequency_filter(padded, butterworth_kernel);
    return result;
}

cv::Mat gaussianfilter::gaussian_high_pass_kernel(cv::Mat scr, float sigma)
{
    cv::Mat gaussianBlur(scr.size(), CV_32FC1); //��CV_32FC1
    float d0 = 2 * sigma * sigma;
    for (int i = 0; i < scr.rows; i++) {
        for (int j = 0; j < scr.cols; j++) {
            float d = pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2);//����,����pow����Ϊfloat��
            gaussianBlur.at<float>(i, j) = 1 - expf(-d / d0);
        }
    }
    /*cv::namedWindow("��˹��ͨ�˲���", cv::WINDOW_FREERATIO);
    imshow("��˹��ͨ�˲���", gaussianBlur);*/
    return gaussianBlur;
}

//��˹��ͨ
cv::Mat gaussianfilter::gaussian_high_pass_filter(cv::Mat& src, float d0)
{
    cv::Mat padded = image_make_border(src);
    cv::Mat gaussian_kernel = gaussian_high_pass_kernel(padded, d0);//�����ͨ�˲���
    cv::Mat result = frequency_filter(padded, gaussian_kernel);
    return result;
}