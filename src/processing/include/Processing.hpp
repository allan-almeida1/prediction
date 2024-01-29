#pragma once
#ifndef DBSCAN_H
#define DBSCAN_h

#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>

class Processing
{
public:
    Processing(std::pair<uint32_t, uint32_t> resolution = std::make_pair<uint32_t, uint32_t>(320, 176), bool verbose = false);
    ~Processing();

    /**
     * @brief Open image from given path
     *
     * @param path The relative or absolute path to the image
     *
     * @return Image
     */
    cv::Mat openImage(const cv::String &path);

    /**
     * @brief Show an image and wait for any key to be pressed to close
     *
     * @param image Image to be displayed
     * @param title Window title
     */
    void showImage(cv::Mat image, cv::String title = "Image");

    /**
     * @brief Convert image to Grayscale and normalize it
     *
     * @param image Image to be preprocessed
     */
    void preprocess(cv::Mat &image);

    /**
     * @brief Binarize the image, return only 0 of 1 values
     *
     * @param image Image to be binarized
     * @param threshold The threshold to use when binarizing the image. Default is 0.6
     */
    void binarize(cv::Mat &image, double threshold = 0.6);

    /**
     * @brief Find the coordinates of active (=1) pixels in the given image
     *
     * @param image Image to search for active pixels
     */
    std::vector<cv::Point> findActivePixels(const cv::Mat &image);

    /**
     * @brief Clusterize the given dataset using DBSCAN algorithm and return the best cluster
     * (where lane lines come far to the image bottom).
     *
     * @param dataset Input dataset
     * @param eps Radius of search (default is 5.0)
     * @param min_points Minimum number of neighbors to consider a point a core point (default is 10U)
     *
     * @return The cluster for each given point. n = 1, 2... or -1 if outlier (no cluster).
     */
    arma::mat dbscan(const std::vector<cv::Point> &dataset, const double eps = 5.0, const size_t min_points = 10UL);

    /**
     * @brief Create a cv::Mat from given arma::mat and show the image.
     *
     * @param cluster Matrix with the list of pixel coordinates to be set in the image (2, N)
     */
    void drawCluster(const arma::mat &cluster);

    /**
     * @brief Fit a second-order polynomial to data and return the coefficients
     *
     * @param data Data to fit
     *
     * @return Eigen::VectorXd containing 3 coefficients [a0 a1 a2]
     */
    Eigen::VectorXd leastSquaresFit(const arma::mat &data);

    /**
     * @brief Calculate the points to be drawn and return the coordinates
     *
     * @param coefficients Vector with the 3 coefficients [a0 a1 a2]
     * @param dataset The arma::mat with the cluster to calculate curve limits
     * @param n_points Number of points to be generated in the vector
     *
     * @return A vector with the coordinates of the points to be drawn
     */
    std::vector<cv::Point> calculateCurve(const Eigen::VectorXd &coefficients, const arma::mat &dataset, const uint16_t &n_points = 10);

    /**
     * @brief Draw the curve along with the original image
     *
     * @param points Vector with the coordinates of the points to be drawn
     * @param image Original image
     */
    void drawCurve(const std::vector<cv::Point> &points, cv::Mat &image);

    /**
     * @brief Start timer
     */
    void startTimer();

    /**
     * @brief Stop timer
     *
     * @param description Description of operation being timed
     */
    void stopTimer(const char *description = "");

private:
    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
    std::pair<uint32_t, uint32_t> resolution;
    bool VERBOSE;
};

#endif // DBSCAN_H