#include "Processing.hpp"
#include <ros/ros.h>
#include <gtest/gtest.h>
#include <ros/package.h>
#include <iostream>

std::vector<cv::Point> g_active_pixels;

class ProcessingTest : public ::testing::Test
{
public:
protected:
    Processing *processing;

    void SetUp() override
    {
        processing = new Processing();
    }

    void TearDown() override
    {
        delete processing;
    }
};

TEST_F(ProcessingTest, PreprocessImage)
{
    ROS_INFO("Open image");
    std::string image_path = ros::package::getPath("prediction") + "/tests/test_image.jpg";
    cv::Mat image = processing->openImage(image_path);

    ASSERT_EQ(image.rows, 176);
    ASSERT_EQ(image.cols, 320);
    ASSERT_FALSE(image.empty());
    ASSERT_EQ(image.channels(), 3);

    ROS_INFO("Preprocess image");
    processing->preprocess(image);

    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(image, &min_val, &max_val, &min_loc, &max_loc);

    ASSERT_EQ(image.channels(), 1);
    ASSERT_LE(max_val, 1.0);
    ASSERT_GE(min_val, 0);

    ROS_INFO("Binarize image");
    processing->binarize(image);
    for (int row = 0; row < image.rows; ++row)
    {
        for (int col = 0; col < image.cols; ++col)
        {
            uchar pixel_value = image.at<uchar>(row, col);
            ASSERT_TRUE((pixel_value == 0 || pixel_value == 1.0));
        }
    }

    ROS_INFO("Get active pixels");
    std::vector<cv::Point> active_pixels = processing->findActivePixels(image);

    ASSERT_EQ(active_pixels.size(), 2720);

    g_active_pixels = active_pixels;
}

TEST_F(ProcessingTest, LeastSquaresFit)
{
    ROS_INFO("Fit polynomial using Least-squares method");
    Eigen::VectorXd coefficients = processing->leastSquaresFit(g_active_pixels, 2);

    ASSERT_EQ(coefficients.size(), 3);

    ROS_INFO("Calculate curve");
    std::vector<cv::Point> points = processing->calculateCurve(coefficients, 9);

    ASSERT_EQ(points.size(), 9);
}

TEST_F(ProcessingTest, RansacFit)
{
    ROS_INFO("Fit polynomial using RANSAC with Least-squares");
    Eigen::VectorXd coefficients = processing->ransacFit(g_active_pixels, 5, 6, 10, 300);

    ASSERT_EQ(coefficients.size(), 6);

    ROS_INFO("Calculate curve");
    std::vector<cv::Point> points = processing->calculateCurve(coefficients, 15);

    ASSERT_EQ(points.size(), 15);
}