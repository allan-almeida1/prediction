#!/usr/bin/env python3

"""
Created on Sat Feb 17 2024
Author: Allan Souza Almeida
Email: allan.almeida@ufba.br
"""

import rospy
import rospkg
from geometry_msgs.msg import Pose
from datetime import datetime
import json

class GroundTruth:
    """
    Ground Truth node class

    This class is responsible for subscribing to the ground truth topic and
    saving the data to a file.
    """

    def __init__(self):
        """
        Initialize ground truth node
        """
        
        rospy.loginfo("Running ground truth node...")

        # Initialize variables
        self.topic_name = "/unity/husky/pose"
        self.path = rospkg.RosPack().get_path('prediction') + "/src/ground_truth/"
        self.pose_sub = rospy.Subscriber(self.topic_name, Pose, self.pose_callback)
        # Current time stamp as string
        current_datetime = datetime.now()
        timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f"{self.path}ground_truth_{timestamp_str}.json"
        self.data = []
        rospy.loginfo("Ground truth node ready")

    def save_data(self):
        """
        Save data to file
        """
        with open(self.filename, 'w') as file:
            json.dump(self.data, file, indent=4)
        rospy.loginfo(f"Data saved to {self.filename}")


    def pose_callback(self, data: Pose):
        """
        Callback function for ground truth subscriber
        """
        rospy.loginfo(f"Received pose data: {data}")
        pose_data = {
            "position": {
                "x": data.position.x,
                "y": data.position.y,
                "z": data.position.z
            },
            "orientation": {
                "x": data.orientation.x,
                "y": data.orientation.y,
                "z": data.orientation.z,
                "w": data.orientation.w
            }
        }

        self.data.append(pose_data)


def main():
    rospy.init_node('ground_truth')
    ground_truth = GroundTruth()
    rospy.spin()
    ground_truth.save_data()


if __name__ == '__main__':
    main()