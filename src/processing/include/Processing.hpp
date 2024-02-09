/**
 * @author Allan Souza Almeida
 * @date Jan 31 2024
 */

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <prediction/Path.h>

/**
 * @brief This namespace implements several methods to process a binary image containing lane lines
 * such as binarize, preprocess, ransacFit, leastSquaresFit etc
 */
namespace processing
{
    /**
     * @brief Image resolution
     */
    struct Resolution
    {
        int16_t width;  // Image width (x dimension)
        int16_t height; // Image height (y dimension)

        Resolution() : width(0), height(0) {}
        Resolution(int16_t w, int16_t h) : width(w), height(h) {}
    };

    /**
     * @brief Polynomial curve
     */
    struct Curve
    {
        int16_t y_min;                 // Minimum y-axis limit to plot the curve
        int16_t y_max;                 // Maximum y-axis limit to plot the curve
        std::vector<cv::Point> points; // A vector with the coordinates of the points to be ploted
    };

    /**
     * @brief Open image from given path
     *
     * @param path The relative or absolute path to the image
     *
     * @return Image
     */
    cv::Mat
    openImage(const cv::String &path);

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
     * @param resolution Image resolution
     */
    void drawCluster(const arma::mat &cluster, Resolution resolution);

    /**
     * @brief Fit an n-order polynomial to data points and return the coefficients
     *
     * @param coordinates A vector with coordinates (x, y) of white pixels
     * @param order Order of the polynomial
     *
     * @return Eigen::VectorXd containing 3 coefficients [a0 a1 a2]
     */
    Eigen::VectorXd leastSquaresFit(const std::vector<cv::Point> &coordinates, const int &order);

    /**
     * @brief Fit an n-order polynomial to data points using RANSAC algorithm and return the coefficients
     *
     * @param coordinates A vector with cordinates (x, y) of white pixels
     * @param order Order of the polynomial
     * @param min_samples Minimum number of samples chosen randomly from data
     * @param threshold Maximum distance for a sample to be considered an inlier
     * @param max_iterations Maximum number of iterations for random sample selection
     *
     * @return Eigen::VectorXd containing 3 coefficients [a0 a1 a2]
     */
    Eigen::VectorXd ransacFit(const std::vector<cv::Point> &coordinates, const int &order, const int &min_samples, const int &threshold, const int &max_iterations);

    /**
     * @brief Calculate the points to be drawn and return the coordinates
     *
     * @param coefficients Vector with the `n+1` coefficients [a0 a1 a2 ... an]
     * @param resolution Image resolution
     * @param n_points Number of points to be generated in the vector (defaults to 9)
     *
     * @return A Curve object that holds the coordinates of the points to be ploted and the y-limits (min/max)
     */
    Curve calculateCurve(const Eigen::VectorXd &coefficients, Resolution resolution, const uint16_t &n_points = 9);

    /**
     * @brief Draw the curve along with the original image
     *
     * @param curve A Curve object that holds the coordinates of the points to be ploted and the y-limits (min/max)
     * @param image Original image
     */
    void drawCurve(const Curve &curve, cv::Mat &image);

    /**
     * @brief Get the last path.
     *
     * @return prediction::Path containing image resolution, polynomial coefficients and y limits to plot the path
     */
    prediction::Path getPath();

    /**
     * @brief Calculates the error between data points and the estimated polynomial and find inliers
     *
     * @param coordinates A vector with data points (x, y)
     * @param model Polynomial coefficients [a0 a1 a2]
     * @param threshold Maximum distance for a sample to be considered an inlier
     *
     * @returns Vector with inliers
     */
    std::vector<cv::Point> findInliers(const std::vector<cv::Point> &coordinates, const Eigen::VectorXd &model, const int &threshold);

}; // namespace processing