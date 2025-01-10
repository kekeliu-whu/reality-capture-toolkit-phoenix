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
    # rosbag 文件路径
    bag_path = '/media/rick/0CFE6A5BAFA76FD0/BaiduNetdiskDownload/2024-12-04-11-20-16-SHAREUAV-S20-outdoor/all_2024-12-04-11-20-16.bag'
    
    # 创建保存图片的根目录
    root_dir = os.path.dirname(bag_path)  # 请替换为实际的保存路径
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # 读取 rosbag 文件
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/camera_agent/img_left/compressed', '/camera_agent/img_right/compressed']):
            # 获取时间戳
            timestamp = msg.header.stamp.to_sec()
            # 将时间戳转换为字符串格式，精确到微秒
            timestamp_str = '{:.9f}'.format(timestamp)
            # 构建文件名
            file_name = f'{timestamp_str}.jpg'
            # 根据相机主题创建不同的文件夹
            if topic == '/camera_agent/img_left/compressed':
                camera_dir = os.path.join(root_dir, 'colmap/images/left_camera')
            elif topic == '/camera_agent/img_right/compressed':
                camera_dir = os.path.join(root_dir, 'colmap/images/right_camera')
             
            if not os.path.exists(camera_dir):
                os.makedirs(camera_dir)
            
            # 保存图像
            file_path = os.path.join(camera_dir, file_name)
            save_image(msg, file_path)
            print(f'Image saved: {file_path}')

if __name__ == '__main__':
    main()
