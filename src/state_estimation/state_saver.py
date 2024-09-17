#!/usr/bin/env python3

"""
Created on Sun Mar 28 2024
Author: Allan Souza Almeida
Email: allan.almeida@ufba.br
"""

import rospy
import rospkg
import json
import time
from opencv101.msg import desvioParams


class StateSaver:
    """
    Class to save state to a file
    """

    def __init__(self, file_path: str):
        """
        Initialize StateSaver class

        Args:
            file_path (str): File path to save state
        """
        self.file_path = file_path
        self.states = []
        rospy.loginfo(f"Subscribing to /desvio_da_curvatura")
        self.image_sub = rospy.Subscriber(
            "/desvio_da_curvatura", desvioParams, self.states_callback)


    def states_callback(self, data: desvioParams):
        """
        Callback for /desvio_da_curvatura topic

        Args:
            data (desvioParams): desvioParams message
        """
        rospy.loginfo(f"Received data: de0={data.de0}, thetae0={data.thetae0}")
        current_state = {
            "de0": data.de0,
            "thetae0": data.thetae0,
            "timestamp": time.time(),
            "ros_time": rospy.Time.now().to_sec()
        }
        self.states.append(current_state)

    
    def save_states(self):
        """
        Save states to a file
        """
        with open(self.file_path, "w") as file:
            json.dump(self.states, file)
        rospy.loginfo(f"States saved to {self.file_path}")


if __name__ == "__main__":
    rospy.init_node("state_saver")
    file_path = rospy.get_param("~file_path", rospkg.RosPack().get_path("prediction") + "/src/state_estimation/states_allan.json")
    state_saver = StateSaver(file_path)
    rospy.spin()
    state_saver.save_states()
    rospy.loginfo("State Saver finished")