#!/usr/bin/env python3

import rospy
from prediction.msg import States
from geometry_msgs.msg import Twist


class PIController:
    """
    PI Controller node class
    """

    def __init__(self):
        """
        Initialize PI Controller node
        """
        self.kp_z = -5
        self.kp_theta = -0.5
        self.ki_z = -0.45
        self.integral_z = 0
        self.last_time = None
        self.linear_velocity = 0.15
        self.limit_angular_velocity = 1.2

        rospy.loginfo("Running PI Controller node...")

        rospy.loginfo(f"Subscribing to /desvio_da_curvatura")

        self.image_sub = rospy.Subscriber(
            "/desvio_da_curvatura", States, self.states_callback)

        self.cmd_vel_pub = rospy.Publisher(
            "/cmd_vel", Twist, queue_size=10)

        rospy.loginfo("PI Controller node ready")

    def states_callback(self, data: States):
        """
        Callback for /desvio_da_curvatura topic

        Args:
            data (States): States message
        """
        # rospy.loginfo(f"Desvio da curvatura: {data}")
        self.control(data)

    def control(self, data: States):
        """
        Control the robot based on the curvature deviation

        Args:
            data (States): States message
        """
        error_z = data.de0
        error_theta = data.thetae0
        control_signal = self.kp_theta * error_theta + self.kp_z * error_z
        if self.last_time is not None:
            dt = rospy.get_time() - self.last_time
            self.integral_z += error_z * dt
            control_signal += self.ki_z * self.integral_z
        self.last_time = rospy.get_time()
        control_signal = max(
            min(control_signal, self.limit_angular_velocity), -self.limit_angular_velocity)
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
