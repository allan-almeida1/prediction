#!/usr/bin/env python3

import rospy
from opencv101.msg import desvioParams
from geometry_msgs.msg import Twist

class PIController:
    """
    PI Controller node class
    """

    def __init__(self):
        """
        Initialize PI Controller node
        """
        self.kp = -5
        self.ki = 0.1
        self.linear_velocity = 0.1

        rospy.loginfo("Running PI Controller node...")

        rospy.loginfo(f"Subscribing to /desvio_da_curvatura")

        self.image_sub = rospy.Subscriber(
            "/desvio_da_curvatura", desvioParams, self.states_callback)
        
        self.cmd_vel_pub = rospy.Publisher(
            "/cmd_vel", Twist, queue_size=10)

        rospy.loginfo("PI Controller node ready")

    def states_callback(self, data: desvioParams):
        """
        Callback for /desvio_da_curvatura topic

        Args:
            data (desvioParams): desvioParams message
        """
        # rospy.loginfo(f"Desvio da curvatura: {data}")
        self.control(data)

    def control(self, data: desvioParams):
        """
        Control the robot based on the curvature deviation

        Args:
            data (desvioParams): desvioParams message
        """
        error_z = data.de0
        error_theta = data.thetae0
        control_signal = self.kp * error_z + 0.1 * error_theta
        rospy.loginfo(f"Control signal: {control_signal}")
        twist = Twist()
        twist.linear.x = self.linear_velocity
        twist.angular.z = control_signal
        self.cmd_vel_pub.publish(twist)
        

    def run(self):
        """
        Run PI Controller node
        """
        rospy.spin()

    def exit(self):
        """
        Exit PI Controller node
        """
        rospy.loginfo("Shutting down PI Controller node...")
        rospy.signal_shutdown("Shutdown")


if __name__ == "__main__":
    rospy.init_node("pi_controller_node")
    pi_controller = PIController()
    pi_controller.run()

