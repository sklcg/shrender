#pragma once

#include <opencv2/opencv.hpp>

#include "sh/default_image.h"
#include "sh/spherical_harmonics.h"

#define M_PI 3.14159265358979323846

std::vector<float> EvalSH(float theta, float phi, unsigned int order = 4){
    std::vector<float> coeffs;

    for (int l = 0; l < order; l++) {
        for (int m = -l; m <= l; m++) {
            coeffs.push_back(sh::EvalSH(l, m, phi, theta));
        }
    }
    return coeffs;
}

// image conversion between OpenCV and SH
sh::DefaultImage convert(const cv::Mat3f& cvImg){
    sh::DefaultImage result(cvImg.cols, cvImg.rows);

    #pragma omp parallel for
    for (int row = 0; row < cvImg.rows; row++){
        for (int col = 0; col < cvImg.cols; col++){
            Eigen::Array3f pixel;
            pixel[0] = cvImg(row, col)[0];
            pixel[1] = cvImg(row, col)[1];
            pixel[2] = cvImg(row, col)[2];

            result.SetPixel(col, row, pixel);
        }
    }
    return result;
}

cv::Mat3f convert(const sh::DefaultImage& shImg){
    cv::Mat3f cvImg(shImg.height(), shImg.width());

    #pragma omp parallel for
    for (int row = 0; row < cvImg.rows; row++){
        for (int col = 0; col < cvImg.cols; col++){
            Eigen::Array3f pixel = shImg.GetPixel(col, row);
            cvImg(row, col) = cv::Vec3f(pixel[0], pixel[1], pixel[2]);
        }
    }
    return cvImg;
}

float halton(int idx, int base){
    float f = 1.f;
    float r = 0.f;
    while (idx > 0){
        f = f / base;
        r = r + f * (idx % base);
        idx = idx / base;
    }
    return r;
}

cv::Vec3f uniform_mapping(float u, float v){
    float phi = v * 2.0 * M_PI;

    //- float cosTheta = 1.0 - u;
    float cosTheta = 1.0 - 1 * u;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return cv::Vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

cv::Vec3f lookup(const cv::Mat3f& img, float x, float y){
    cv::Mat3f patch;
    cv::getRectSubPix(img, cv::Size(1, 1), cv::Point2f(x, y), patch);
    return patch(0, 0);
}

cv::Vec2f to_sphere(float x, float y, float z){
    cv::Vec2f vec;

    vec[0] = acosf(y);
    vec[1] = atan2f(-x, z) + M_PI;

    return vec;
}

cv::Vec2f to_sphere(const cv::Vec3f& vec){
    return to_sphere(vec[0], vec[1], vec[2]);
}

cv::Vec3f to_vector(float theta, float phi){
    cv::Vec3f vec;

    float r = sin(theta);
    vec[1] = cos(theta);
    vec[0] = r * sin(phi);
    vec[2] = -r * cos(phi);

    return vec;
}

cv::Vec3f to_vector(const cv::Vec2f& sphere){
    return to_vector(sphere[0], sphere[1]);
}

float to_degree(float rad){
    return rad * 180.f / M_PI;
}

float to_radius(float deg){
    return deg / 180.f * M_PI;
}

cv::Mat3f rotate(const cv::Mat3f & envmap, float t, float p){
    // get rotation quaternion
    Eigen::Quaterniond q;
    q = Eigen::AngleAxisd(-p, Eigen::Vector3d(0, 0, 1))
        * Eigen::AngleAxisd(-t, Eigen::Vector3d(0, 1, 0));
    q.normalize();

    // result environment map
    cv::Mat3f result(envmap.size());

    #pragma omp parallel for
    for (int x = 0; x < result.cols; x++){
        for (int y = 0; y < result.rows; y++){
            double theta = sh::ImageYToTheta(y, result.rows);
            double phi = sh::ImageXToPhi(x, result.cols);

            Eigen::Vector3d dir = sh::ToVector(phi, theta);

            sh::ToSphericalCoords(q * dir, &phi, &theta);

            Eigen::Vector2d v2d = sh::ToImageCoords(phi, theta, result.cols, result.rows);

            //printf("%d, %d, %f, %f\n", x, y, v2d[0], v2d[1]);

            result(y, x) = lookup(envmap, v2d[0] - 0.5f, v2d[1] - 0.5f);
        }
    }
    return result;
}

cv::Vec2f getRotation(const cv::Vec3f& v1, const cv::Vec3f& v2) {
    return to_sphere(cv::normalize(v1)) - to_sphere(cv::normalize(v2));
}
cv::Vec2f getRotation(float x1, float y1, float z1, float x2, float y2, float z2) {
    cv::Vec2f vec1 = to_sphere(cv::normalize(cv::Vec3f(x1, y1, z1)));
    cv::Vec2f vec2 = to_sphere(cv::normalize(cv::Vec3f(x2, y2, z2)));
    return (vec1 - vec2);
}