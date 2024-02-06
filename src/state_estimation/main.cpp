#include <ros/ros.h>
#include "StateEstimationNode.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "state_estimation");
    ros::NodeHandle nh;
    StateEstimationNode stateEstimationNode(&nh);
    ros::spin();
    return 0;
}