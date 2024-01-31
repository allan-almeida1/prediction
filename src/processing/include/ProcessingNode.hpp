/**
 * @author Allan Souza Almeida
 * @date Jan 31 2024
 */

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <deque>
#include "Processing.hpp"

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
    ros::Subscriber img_sub; // Image topic subscriber
    Processing processing;
    uint16_t window_size;                       // Window size for moving average calculation (filter RANSAC)
    std::deque<Eigen::VectorXd> ransac_results; // Deque containing last RANSAC results
};