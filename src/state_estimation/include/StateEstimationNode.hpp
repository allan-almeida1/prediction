/**
 * @author Allan Souza Almeida
 * @date Feb 5 2024
 */

#pragma once

#ifndef STATE_ESTIMATION_NODE_HPP
#define STATE_ESTIMATION_NODE_HPP

#include <ros/ros.h>
#include <prediction/Path.h>
#include "StateEstimation.hpp"

/**
 * @brief This class is responsible for estimating the model states from the curve that was fit to the lane
 */
class StateEstimationNode
{
public:
    StateEstimationNode(ros::NodeHandle *nh);
    ~StateEstimationNode();

    /**
     * @brief Receive a Path object from `/prediction/path` topic
     */
    void pathCallback(const prediction::Path::Ptr &path);

private:
    ros::Subscriber path_sub;         // Subscriber for the `/prediction/path` topic
    StateEstimation state_estimation; // Instance of StateEstimation class
};

#endif // STATE_ESTIMATION_NODE_HPP