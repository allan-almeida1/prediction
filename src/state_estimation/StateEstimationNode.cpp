/**
 * @author Allan Souza Almeida
 * @date Feb 5 2024
 */

#include <StateEstimationNode.hpp>

StateEstimationNode::StateEstimationNode(ros::NodeHandle *nh)
{
    this->path_sub = nh->subscribe("/prediction/path", 100, &StateEstimationNode::pathCallback, this);
    this->state_estimation = StateEstimation(0.0349, 0, 0.68068 / 151);
}

StateEstimationNode::~StateEstimationNode()
{
}

void StateEstimationNode::pathCallback(const prediction::Path::Ptr &path)
{
    double theta = state_estimation.getAngle(path, AngleUnit::DEGREES);
    double z = state_estimation.getLateralDisplacement(path);
    ROS_INFO("Theta: %.2f deg | Z: %d px", theta, int(z));
}