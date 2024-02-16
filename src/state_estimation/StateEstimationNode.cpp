/**
 * @author Allan Souza Almeida
 * @date Feb 5 2024
 */

#include <StateEstimationNode.hpp>

StateEstimationNode::StateEstimationNode(ros::NodeHandle *nh)
{
    this->path_sub = nh->subscribe("/prediction/path", 100, &StateEstimationNode::pathCallback, this);
    this->params_pub = nh->advertise<opencv101::desvioParams>("/desvio_da_curvatura", 100);
    nh->param("/state_estimation/Z_cal", this->Z_cal, 1.0);
    nh->param("/state_estimation/theta_cal", this->theta_cal, 1.0);
}

StateEstimationNode::~StateEstimationNode()
{
}

void StateEstimationNode::pathCallback(const prediction::Path::Ptr &path)
{
    States states = state_estimation::getStates(path, this->Z_cal, this->theta_cal, AngleUnit::RADIANS);
    double curvature = state_estimation::calculateCurvature(path, this->Z_cal);
    ROS_INFO("Theta: %.3f rad | Z: %.3f m | Curvature: %.3f", states.theta, states.Z, curvature);
    opencv101::desvioParams params;
    params.cmax = 0;
    params.curvature = curvature;
    params.de0 = states.Z;
    params.thetae0 = states.theta;
    params.exectime = 0;
    params.eta0 = 0;
    this->params_pub.publish(params);
}