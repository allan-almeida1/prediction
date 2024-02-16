/**
 * @author Allan Souza Almeida
 * @date Jan 31 2024
 */

#pragma once

#ifndef PROCESSING_NODE_HPP
#define PROCESSING_NODE_HPP

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <deque>
#include "Processing.hpp"
#include "ExecTime.hpp"

/**
 * @brief This class is responsible for processing a binary image coming from a topic
 */
class ProcessingNode
{
public:
    ProcessingNode(ros::NodeHandle *nh);
    ~ProcessingNode();

    /**
     * @brief Receive image from /image_raw_bin
     *
     * @param img Received image
     */
    void imageCallback(const sensor_msgs::Image::Ptr &img);

private:
    ros::NodeHandle *nh;
    ros::Subscriber img_sub;                    // Image topic subscriber
    ros::Publisher img_pub;                     // Path img publisher
    ros::Publisher path_pub;                    // Path publisher
    ExecTime execTime;                          // Used to calculate execution time
    int window_size;                            // Window size for moving average calculation (filter RANSAC)
    int order;                                  // Polynomial order for RANSAC fitting
    int min_samples;                            // Minimum number of samples for RANSAC
    int threshold;                              // Maximum threshold for a sample to be considered an inlier
    int max_iterations;                         // Maximum number of iterations for RANSAC
    int n_points;                               // Number of points to draw the curve
    int width;                                  // Width of the image
    int height;                                 // Height of the image
    bool publish_image;                         // Wether to publish the resulting image
    bool show_image;                            // Wether to show the resulting image
    processing::Resolution resolution;          // Image resolution
    std::deque<Eigen::VectorXd> ransac_results; // Deque containing last RANSAC results
};

#endif // PROCESSING_NODE_HPP