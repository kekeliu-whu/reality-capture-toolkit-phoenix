#!/usr/bin/env python

import os
import rospy
import rosbag
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from datetime import datetime

def save_image(compressed_img_msg, file_path):
    np_arr = np.fromstring(compressed_img_msg.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imwrite(file_path, image)

def main():
    # Path to rosbag file
    bag_path = '/media/rick/0CFE6A5BAFA76FD0/BaiduNetdiskDownload/2024-12-04-11-20-16-SHAREUAV-S20-outdoor/all_2024-12-04-11-20-16.bag'
    
    # Create root directory for saving images
    root_dir = os.path.dirname(bag_path)  # Replace with actual output path if needed
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # Read rosbag file
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/camera_agent/img_left/compressed', '/camera_agent/img_right/compressed']):
            # Get timestamp
            timestamp = msg.header.stamp.to_sec()
            # Convert timestamp to string format with microsecond precision
            timestamp_str = '{:.9f}'.format(timestamp)
            # Build file name
            file_name = f'{timestamp_str}.jpg'
            # Create separate folders based on camera topic
            if topic == '/camera_agent/img_left/compressed':
                camera_dir = os.path.join(root_dir, 'colmap/images/left_camera')
            elif topic == '/camera_agent/img_right/compressed':
                camera_dir = os.path.join(root_dir, 'colmap/images/right_camera')
             
            if not os.path.exists(camera_dir):
                os.makedirs(camera_dir)
            
            # Save image
            file_path = os.path.join(camera_dir, file_name)
            save_image(msg, file_path)
            print(f'Image saved: {file_path}')

if __name__ == '__main__':
    main()
