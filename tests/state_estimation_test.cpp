#include "StateEstimation.hpp"
#include "Processing.hpp"
#include <prediction/Path.h>
#include <gtest/gtest.h>
#include <ros/ros.h>
#include <ros/package.h>

prediction::Path::Ptr getPathFromImage(const std::string image_path)
{
    cv::Mat image = processing::openImage(image_path);
    processing::preprocess(image);
    processing::binarize(image);
    std::vector<cv::Point> coordinates = processing::findActivePixels(image);
    Eigen::VectorXd coefficients = processing::ransacFit(coordinates, 3, 4, 10, 200);
    processing::Curve curve = processing::calculateCurve(coefficients, processing::Resolution(320, 176));
    prediction::Path path;
    std::vector<double> coefficients_v(coefficients.data(), coefficients.data() + coefficients.size());
    path.coefficients = coefficients_v;
    path.limits.y_min = curve.y_min;
    path.limits.y_max = curve.y_max;
    path.resolution.width = 320;
    path.resolution.height = 176;
    boost::shared_ptr<prediction::Path> path_ptr = boost::make_shared<prediction::Path>(path);
    return path_ptr;
}

TEST(StateEstimationTest, GetStates)
{
    std::string image_path = ros::package::getPath("prediction") + "/tests/test_z_image.jpg";
    prediction::Path::Ptr path_ptr = getPathFromImage(image_path);
    States states = state_estimation::getStates(path_ptr, 0.006622517, 1.51, AngleUnit::DEGREES);
    ROS_INFO("Z: %.2f m, Theta: %2f deg", states.Z, states.theta);
    ASSERT_LE(states.Z, -0.69);
    ASSERT_GE(states.Z, -0.8);
    ASSERT_GE(states.theta, 34);
    ASSERT_LE(states.theta, 37);
}

TEST(StateEstimationTest, CalculateCurvature)
{
    std::string image_path = ros::package::getPath("prediction") + "/tests/test_curvature_image.jpg";
    prediction::Path::Ptr path_ptr = getPathFromImage(image_path);
    double curvature = state_estimation::calculateCurvature(path_ptr, 0.006622517);
    ROS_INFO("Curvature: %.3f", curvature);
    ASSERT_GE(curvature, 8.0);
    ASSERT_LE(curvature, 10);
}