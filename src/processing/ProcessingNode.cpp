#include "ProcessingNode.hpp"

ProcessingNode::ProcessingNode(ros::NodeHandle *nh)
{
    this->processing = Processing();
    this->img_sub = nh->subscribe("/image_raw_bin", 100, &ProcessingNode::imageCallback, this);
    this->window_size = 5;
}

ProcessingNode::~ProcessingNode() {}

void ProcessingNode::imageCallback(const sensor_msgs::Image::Ptr &img)
{
    processing.startTimer();

    cv::Mat image;
    image = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8)->image;

    cv::resize(image, image, cv::Size(320, 176), 0, 0, CV_INTER_LINEAR);

    cv::Mat original_img = image;

    // Preprocessing step
    processing.preprocess(image);

    // Processing step
    auto start = std::chrono::high_resolution_clock::now();

    processing.binarize(image);

    std::vector<cv::Point> coordinates = processing.findActivePixels(image);

    if (coordinates.size() != 0)
    {
        // Fit a second order polynomial to data using RANSAC
        Eigen::VectorXd coefficients = processing.ransacFit(coordinates, 3, 10, 200);

        // Append coefficients to vector for average calculation
        ransac_results.push_back(coefficients);

        if (ransac_results.size() > window_size)
        {
            ransac_results.pop_front();
        }

        // Calculate average
        Eigen::VectorXd filtered_coefficients = Eigen::VectorXd::Zero(3);
        for (uint16_t i = 0; i < ransac_results.size(); ++i)
        {
            filtered_coefficients += ransac_results[i];
        }
        filtered_coefficients /= ransac_results.size();

        // Calculate points to draw the curve
        std::vector<cv::Point> points = processing.calculateCurve(filtered_coefficients, 8);

        processing.stopTimer("Processing step");

        processing.drawCurve(points, original_img);
    }
    else
    {
        processing.stopTimer("Processing step");
    }
}