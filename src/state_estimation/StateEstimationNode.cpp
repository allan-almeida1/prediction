/**
 * @author Allan Souza Almeida
 * @date Feb 5 2024
 */

#include <StateEstimationNode.hpp>

StateEstimationNode::StateEstimationNode(ros::NodeHandle *nh)
{
    this->path_sub = nh->subscribe("/prediction/path", 100, &StateEstimationNode::pathCallback, this);
    nh->param("/state_estimation/Z_cal", this->Z_cal, 1.0);
    nh->param("/state_estimation/theta_cal", this->theta_cal, 1.0);
}

StateEstimationNode::~StateEstimationNode()
{
}

void StateEstimationNode::pathCallback(const prediction::Path::Ptr &path)
{
    States states = state_estimation::getStates(path, this->Z_cal, this->theta_cal, AngleUnit::DEGREES);
    // TODO: create rosmsg in format expected by the controller
    ROS_INFO("Theta: %.2f deg | Z: %.3f m", states.theta, states.Z);
}