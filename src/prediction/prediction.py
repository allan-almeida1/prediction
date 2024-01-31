#!/usr/bin/env python3

"""
Created on Sun Dec 31 2023
Author: Allan Souza Almeida
Email: allan.almeida@ufba.br
"""

import rospy
import tensorflow as tf
from sensor_msgs.msg import Image
import sys
import os
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.erfnet import ERFNet
from util.util import freeze_model


class Prediction:
    """
    Prediction node class
    """
    
    def __init__(self):
        """
        Initialize prediction node
        """

        rospy.loginfo("Running prediction node...")

        # Initialize variables
        self.resolution = (320, 176)
        self.topic_name = rospy.get_param("/prediction_node/topic_name", "/camera/image")
        rospy.loginfo(f"Subscribing to {self.topic_name}")
        self.bridge = CvBridge()
        
        self.check_gpu()
        self.load_model()
        self.frozen_model = freeze_model(self.model)

        self.image_pub = rospy.Publisher(
            "/image_raw_bin", Image, queue_size=10)
        self.image_sub = rospy.Subscriber(
            self.topic_name, Image, self.image_callback)
        
        rospy.loginfo("Prediction node ready")
        
        
    def check_gpu(self):
        """
        Check if GPU is available
        """

        gpus_available = tf.config.list_physical_devices('GPU')
        if (len(gpus_available) > 0):
            rospy.logwarn("Running on GPU")
        else:
            rospy.logerr("Running on CPU")
        

    def load_model(self):
        """
        Load ERFNet model
        """

        rospy.loginfo("Loading model...")
        erf = ERFNet(input_shape=(176, 320, 3))
        self.model = erf.build()
        self.model.load_weights(os.path.join(
            parentdir, "../weights/20231018_after_ramp.h5"))
        rospy.loginfo("Model loaded")


    def predict_image(self, img: np.ndarray):
        """
        Predict image

        Args:
            img (np.ndarray): image to predict
        """
            
        initial_time = time.time()
        image = cv2.resize(img, self.resolution) / 255
        image = np.array(image, dtype=np.float32)
        input_tensor = np.expand_dims(image, axis=0)
        input_tensor = tf.convert_to_tensor(input_tensor)
        mask_image = self.frozen_model(input_tensor)[0][0]
        rospy.loginfo(f'Prediction time (ms): {round((time.time() - initial_time)*1000, 4)}')
        mask_np = mask_image.numpy()
        mask_np = np.uint8(mask_np * 255)
        mask_np = cv2.resize(mask_np, (640, 360))
        img_bin = self.bridge.cv2_to_imgmsg(mask_np, encoding="mono8")
        self.image_pub.publish(img_bin)
        cv2.imshow("Prediction", mask_np)
        cv2.waitKey(1)


    def image_callback(self, msg: Image):
        """
        Image topic callback
        
        Args:
            msg (Image): image message
        """

        rospy.loginfo("Image received")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # show image
        # cv2.imshow("Image", cv_image)
        # cv2.waitKey(1)
        self.predict_image(cv_image)

        


if __name__ == '__main__':
    rospy.init_node('prediction')
    prediction = Prediction()
    rospy.spin()
