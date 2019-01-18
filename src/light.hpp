#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "sh/spherical_harmonics.h"
#include "sh/default_image.h"

#include "util.hpp"

struct SHLight{
    unsigned int order;
    std::vector<cv::Vec3f> coeffs;

    void GetFromEnvmap(const cv::Mat3f & envmap, unsigned int order = 4){
        this->order = order;
        sh::DefaultImage shEnvmap = convert(envmap);

        // get SH illumination
        std::unique_ptr<std::vector<Eigen::Array3f>> sh = ProjectEnvironment(order, shEnvmap);

        coeffs.clear();
        for (int i = 0; i < sh->size(); i++){
            Eigen::Array3f vec = (*sh)[i];
            coeffs.push_back(cv::Vec3f(vec[0], vec[1], vec[2]));
        }
    }

    void GetFromFile(const std::string& filename) {
        std::ifstream fin(filename, std::ios::in);
        float r, g, b;
        coeffs.clear();
        while (fin >> r >> g >> b) {
            coeffs.push_back(cv::Vec3f(b, g, r));
        }
        order = (int)(sqrt(coeffs.size() / 3 + 1.5f) - 0.1f);
    }

    cv::Vec3f Render(const std::vector<float>& coeff) const{
        cv::Vec3f result(0.f, 0.f, 0.f);

        unsigned int size = cv::min(coeff.size(), this->coeffs.size());

        for (int i = 0; i < size; i++){
            result += this->coeffs[i] * coeff[i];
        }
        for (int i = 0; i < 3; ++i) {
            result[i] = cv::max(result[i], 0.f);
        }
        return result;
    }

    cv::Mat3f SHLight::RenderEnvmap(int width, int height) const{
        sh::DefaultImage shEnvmap(width, height);

        std::vector<Eigen::Array3f> sh;

        for (int i = 0; i < coeffs.size(); i++){
            Eigen::Array3f vec(coeffs[i][0], coeffs[i][1], coeffs[i][2]);
            sh.push_back(vec);
        }

        RenderDiffuseIrradianceMap(sh, &shEnvmap);

        return convert(shEnvmap);
    }

    void Print() const{
        std::cout << "SH Light with order [" << order << "]:" << std::endl;
        for (int i = 0; i < coeffs.size(); i++){
            std::cout << "  " << coeffs[i] << std::endl;
        }
    }
    void Write(const std::string& filepath) const{
        std::ofstream fout(filepath, std::ios::out);
        for (int i = 0; i < coeffs.size(); ++i) {
            cv::Vec3f c = coeffs[i];
            fout << c[2] << " " << c[1] << " " << c[0] << std::endl;
        }
    }
    void WriteXML(const std::string& filepath) const {
        cv::Mat3f sh = cv::Mat3f::zeros(coeffs.size(), 1);
        for (int i = 0; i < coeffs.size(); i++) {
            sh(i, 0) = coeffs[i];
        }
        cv::FileStorage file(filepath, cv::FileStorage::WRITE);
        file << "SH" << sh;
    }
};