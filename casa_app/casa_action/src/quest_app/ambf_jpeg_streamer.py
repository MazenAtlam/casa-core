#!/usr/bin/env python3
"""
ambf_jpeg_streamer.py

Subscribes to an AMBF camera image (sensor_msgs/Image, rgb8, ~0.92 MB per 640x480 frame),
JPEG-compresses it and republishes it as sensor_msgs/CompressedImage for the Quest/Unity app.
The original header (AMBF's wall-clock stamp) is kept so the headset can estimate frame age.

Run:
  ros2 run ... is not needed; just
  python3 ambf_jpeg_streamer.py --ros-args -p quality:=80 -p max_fps:=30.0
"""
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class AmbfJpegStreamer(Node):
    def __init__(self):
        super().__init__('ambf_jpeg_streamer')
        self.declare_parameter('in_topic', '/ambf/env/stereo/left/ImageData')
        self.declare_parameter('out_topic', '/vr/left/image/compressed')
        self.declare_parameter('quality', 80)
        self.declare_parameter('max_fps', 30.0)

        in_topic = self.get_parameter('in_topic').value
        out_topic = self.get_parameter('out_topic').value
        self.quality = int(self.get_parameter('quality').value)
        max_fps = float(self.get_parameter('max_fps').value)
        self.min_period = 1.0 / max_fps if max_fps > 0.0 else 0.0

        # Input: newest frame only. A best-effort subscriber still matches AMBF's reliable publisher.
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        # Output: reliable, because the ROS-TCP-Endpoint's internal subscriber is reliable.
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(CompressedImage, out_topic, pub_qos)
        self.sub = self.create_subscription(Image, in_topic, self.on_image, sub_qos)

        self.last_pub = 0.0
        self.warned_encoding = False
        self.n = 0
        self.bytes = 0
        self.enc_time = 0.0
        self.t_report = time.monotonic()
        self.create_timer(5.0, self.report)
        self.get_logger().info(
            f'{in_topic} -> {out_topic}  (jpeg quality {self.quality}, max {max_fps:.0f} fps)')

    def on_image(self, msg: Image):
        now = time.monotonic()
        if now - self.last_pub < self.min_period:
            return
        if msg.encoding not in ('rgb8', 'bgr8'):
            if not self.warned_encoding:
                self.get_logger().warn(f'unsupported encoding "{msg.encoding}" (need rgb8 or bgr8)')
                self.warned_encoding = True
            return

        h, w = msg.height, msg.width
        buf = np.asarray(msg.data, dtype=np.uint8)
        img = buf.reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3)
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # OpenCV encodes BGR

        ok, jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            self.get_logger().warn('JPEG encoding failed')
            return

        out = CompressedImage()
        out.header = msg.header
        out.format = 'jpeg'
        out.data = jpg.tobytes()
        self.pub.publish(out)

        self.last_pub = now
        self.n += 1
        self.bytes += len(out.data)
        self.enc_time += time.monotonic() - now

    def report(self):
        dt = time.monotonic() - self.t_report
        if self.n:
            self.get_logger().info(
                f'{self.n / dt:.1f} fps | {self.bytes / self.n / 1024:.0f} KB/frame | '
                f'{self.bytes / dt / 1e6:.2f} MB/s | encode {self.enc_time / self.n * 1000:.1f} ms')
        else:
            self.get_logger().warn('no frames received yet (is AMBF running and publishing images?)')
        self.n = 0
        self.bytes = 0
        self.enc_time = 0.0
        self.t_report = time.monotonic()


def main():
    rclpy.init()
    node = AmbfJpegStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
