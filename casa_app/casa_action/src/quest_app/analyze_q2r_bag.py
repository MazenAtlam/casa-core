#!/usr/bin/env python3
"""
analyze_q2r_bag.py -- check a recorded Quest episode (rosbag2, sqlite3) before using it as demonstration data.

Usage:
    source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
    python3 analyze_q2r_bag.py ~/datasets/q2r_test/ep_<timestamp>

Reports, per pose topic: sample rate and gaps by ARRIVAL time (when the PC received it) and by HEADER STAMP
(when the headset sampled it), Wi-Fi jitter (spread of arrival - stamp), position span, quaternion norm.
Per inputs topic: ranges of trigger/grip/stick and how often the buttons were pressed.
"""
import sys

import numpy as np


def describe_gaps(label, t):
    t = np.asarray(t, dtype=float)
    if len(t) < 3 or t[-1] - t[0] <= 0:
        return f"{label}: not enough samples"
    d = np.diff(t)
    return (f"{label}: {len(t) / (t[-1] - t[0]):6.1f} Hz | gap median {np.median(d) * 1e3:5.1f} ms, "
            f"p99 {np.percentile(d, 99) * 1e3:6.1f} ms, max {d.max() * 1e3:6.1f} ms | "
            f"bursts (<2 ms) {np.mean(d < 0.002) * 100:4.1f} %")


def describe_latency(t_arrival, t_stamp):
    lat = (np.asarray(t_arrival, float) - np.asarray(t_stamp, float)) * 1e3
    p5, p50, p95 = np.percentile(lat, [5, 50, 95])
    return (f"arrival - stamp: median {p50:8.1f} ms (includes the Quest/PC clock offset), "
            f"jitter p95-p5 = {p95 - p5:6.1f} ms")


def describe_pose(pos, quat):
    pos = np.asarray(pos, float)
    quat = np.asarray(quat, float)
    span = pos.max(axis=0) - pos.min(axis=0)
    norm_err = np.abs(np.linalg.norm(quat, axis=1) - 1.0).max()
    return (f"position span (m): x {span[0]:.3f}  y {span[1]:.3f}  z {span[2]:.3f} | "
            f"quaternion |norm-1| max {norm_err:.1e}")


def read_bag(path):
    # ROS imports live here so the pure functions above can be tested without ROS.
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    out = {}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        out.setdefault(topic, []).append((t_ns * 1e-9, msg))
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    data = read_bag(sys.argv[1])
    if not data:
        print("The bag is empty: no messages were recorded (is the app connected? did you list the topics?).")
        return 2

    for topic in sorted(data):
        items = data[topic]
        print(f"\n== {topic}  ({len(items)} messages)")
        t_arr = [t for t, _ in items]
        if topic.endswith('_pose'):
            t_st = [m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for _, m in items]
            print("  " + describe_gaps("by arrival", t_arr))
            if max(t_st) > 0:
                print("  " + describe_gaps("by stamp  ", t_st))
                print("  " + describe_latency(t_arr, t_st))
            else:
                print("  header stamps are all 0 (store app): only arrival times are available")
            pos = [[m.pose.position.x, m.pose.position.y, m.pose.position.z] for _, m in items]
            quat = [[m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w]
                    for _, m in items]
            print("  " + describe_pose(pos, quat))
        elif topic.endswith('_inputs'):
            print("  " + describe_gaps("by arrival", t_arr))
            for f in ('press_index', 'press_middle', 'thumb_stick_horizontal', 'thumb_stick_vertical'):
                v = np.array([getattr(m, f) for _, m in items], float)
                print(f"  {f:24s} min {v.min():6.2f}  max {v.max():6.2f}")
            for f in ('button_upper', 'button_lower'):
                v = np.array([bool(getattr(m, f)) for _, m in items])
                print(f"  {f:24s} pressed in {v.mean() * 100:5.1f} % of samples")
        else:
            print("  " + describe_gaps("by arrival", t_arr))
    return 0


if __name__ == '__main__':
    sys.exit(main())
