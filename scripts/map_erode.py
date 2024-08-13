#!/usr/bin/env python3
# coding: utf-8
import cv2
import rospy
import numpy as np
from copy import copy
from nav_msgs.msg import OccupancyGrid
from key_params import params

map_topic = rospy.get_param('~map_topic', '/projected_map_erode')
kernel_size = params['scene1']['erode_kernal_size'] # rospy.get_param('~erode_kernal_size', 3)
kernel_dilate = np.ones((2, 2), np.uint8)
kernel = np.ones((kernel_size, kernel_size), np.uint8)
pub = rospy.Publisher(map_topic, OccupancyGrid, queue_size=10)
new_msg = OccupancyGrid()

OBSTACLE = 100

def map_callback(msg):
    global new_msg, pub, kernel
    # rospy.loginfo(type(msg.data))
    # rospy.loginfo(len(msg.data))
    # rospy.loginfo(msg.data.shape)
    map_array = np.asarray(msg.data).reshape((msg.info.height,
                                        msg.info.width))
    binary = np.zeros_like(map_array, np.uint8)
    binary.fill(255)  # white
    binary[np.where(map_array == OBSTACLE)] = 0  # black
    
    # ret, binary = cv2.threshold(map_array, 0, 255, cv2.THRESH_BINARY_INV)  # 0 if > OBSTACLE else 255
    # cv2.imwrite('binary.png', binary)
    
    # dilate = cv2.dilate(binary, kernel_dilate)   
    erode = cv2.erode(binary, kernel)
     
    # cv2.imwrite('erode.png', erode)
    # erode = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # close is bad
    
    map_array[np.where(erode == 0)] = OBSTACLE # 0 is black
    # map_array[np.where(map_array == -1)] = 205 
    # map_array[np.where(map_array == 0)] = 254 
    # cv2.imwrite('erode_map.png', map_array)
    # assert False

    # new_msg.info.height = msg.info.height
    # new_msg.info.width = msg.info.width
    # new_msg.info.resolution = msg.info.resolution
    # new_msg.info.origin.position.x = msg.info.origin.position.x
    # new_msg.info.origin.position.y = msg.info.origin.position.y
    new_msg.info = msg.info
    new_msg.header = msg.header
    # new_msg.header.frame_id = msg.header.frame_id
    # new_msg.header.stamp = rospy.Time().now()
    new_msg.data = tuple(map_array.ravel()) #.tolist()  # ravel()
    # rospy.loginfo(new_msg.data == msg.data)
    assert len(new_msg.data) == len(msg.data), f'{len(new_msg.data)}, {len(msg.data)}'
    pub.publish(copy(new_msg))
    

if __name__ == '__main__':
    rospy.init_node('map_erode', anonymous=False)
    rateHz = rospy.get_param('~rateHz', 100)
    rate = rospy.Rate(rateHz)
    raw_map_topic = rospy.get_param('~raw_map_topic', '/projected_map')
    rospy.Subscriber(raw_map_topic, OccupancyGrid, map_callback)
    while not rospy.is_shutdown():
        rate.sleep()
    