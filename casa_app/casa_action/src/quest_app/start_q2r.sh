#!/usr/bin/env bash
# Start the ROS-TCP-Endpoint for the Quest (store app or our Unity app, USB or Wi-Fi).
# Usage: ./start_q2r.sh [port]        (default port 10000)
#
# USB mode : app uses 127.0.0.1:PORT, this script runs `adb -d reverse` for you.
# Wi-Fi mode: app uses the LAN IP printed below (PC and headset on the same network).
#
# The endpoint listens on 0.0.0.0 (all interfaces) so BOTH modes work.
# `adb -d` targets the USB-connected headset even when the same headset is also
# listed over Wi-Fi (adb tcpip 5555 + adb connect), where plain `adb` would fail
# with "more than one device/emulator".

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

PORT="${1:-10000}"
IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')

echo "=============================================================="
echo " LAN IP of this PC : ${IP:-unknown}   (Wi-Fi mode: use ${IP:-<ip>}:${PORT} in the app)"

if command -v adb >/dev/null 2>&1 && [ "$(adb -d get-state 2>/dev/null)" = "device" ]; then
  if adb -d reverse "tcp:${PORT}" "tcp:${PORT}" >/dev/null 2>&1; then
    echo " USB headset found : adb reverse set  (USB mode: use 127.0.0.1:${PORT} in the app)"
  else
    echo " USB headset found, but 'adb reverse' failed"
  fi
else
  echo " No USB headset    : Wi-Fi mode only"
fi
echo "=============================================================="

exec ros2 run ros_tcp_endpoint default_server_endpoint \
  --ros-args -p ROS_IP:=0.0.0.0 -p ROS_TCP_PORT:="${PORT}"
