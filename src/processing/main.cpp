#include "Processing.hpp"
#include "ProcessingNode.hpp"
#include <chrono>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "processing_node");
    ros::NodeHandle nh;
    ProcessingNode processingNode(&nh);
    ros::spin();
    return 0;
}