#!/usr/bin/env bash
# quest_link_check.sh -- one-shot diagnosis of the Quest <-> PC link (Wi-Fi or USB).
# Usage: ./quest_link_check.sh [port]      (default port 10000)
# Needs: adb (headset reachable by USB or `adb connect <ip>:5555`), ROS 2 Humble.

PKG=com.adasr.questapp
PORT="${1:-10000}"
hr() { echo; echo "== $* =="; }

hr "PC"
PCIP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
echo "PC LAN IP: ${PCIP:-unknown}   <- this is what the app must use in Wi-Fi mode"

hr "Endpoint listening on :${PORT}?"
if ss -ltn "( sport = :${PORT} )" 2>/dev/null | grep -q LISTEN; then
  ss -ltn "( sport = :${PORT} )" | tail -n +2
else
  echo "NOT listening -> start ./start_q2r.sh"
fi

hr "Headset through adb"
adb devices 2>/dev/null | tail -n +2 | grep -v '^$' || echo "no adb device"
NDEV=$(adb devices 2>/dev/null | awk 'NR>1 && $2=="device"' | wc -l)
SERIAL=$(adb devices 2>/dev/null | awk 'NR>1 && $2=="device" && $1 !~ /:/ {print $1; exit}')   # prefer USB
[ -z "$SERIAL" ] && SERIAL=$(adb devices 2>/dev/null | awk 'NR>1 && $2=="device" {print $1; exit}')
if [ -z "$SERIAL" ]; then ADB="adb"; echo "no headset reachable"; else ADB="adb -s $SERIAL"; fi
if [ "$NDEV" -gt 1 ]; then
  echo "note: $NDEV adb entries (USB + Wi-Fi of the same headset?). Using $SERIAL."
  echo "      Plain 'adb ...' fails with 'more than one device': use 'adb -d' (USB) or 'adb -s <serial>'."
fi
HIP=$($ADB shell ip -f inet addr show wlan0 2>/dev/null | tr -d '\r' | awk '/inet /{print $2}' | cut -d/ -f1)
echo "Headset Wi-Fi IP: ${HIP:-unknown}"
if [ -n "$HIP" ]; then
  ping -c 2 -W 1 "$HIP" 2>/dev/null | tail -n 2 || echo "ping failed"
fi

hr "Clock difference headset - PC (whole seconds)"
HT=$($ADB shell date +%s 2>/dev/null | tr -d '\r'); PT=$(date +%s)
if [ -n "$HT" ]; then
  echo "headset - PC = $((HT - PT)) s   (0 or 1 is fine; the HUD 'age' needs the real offset to be far below 0.1 s)"
else
  echo "unknown"
fi

hr "Installed build of ${PKG}"
$ADB shell dumpsys package "${PKG}" 2>/dev/null | tr -d '\r' | grep -E "lastUpdateTime|firstInstallTime" | sort -u || echo "package not installed"
echo "(lastUpdateTime must be AFTER the moment you pressed Build)"

hr "Address the app has been trying (from the headset log)"
ATT=$($ADB logcat -d -s Unity 2>/dev/null | tr -d '\r' | grep -oE "Connection to [0-9.]+:[0-9]+ failed" | sort | uniq -c | tail -n 3)
if [ -n "$ATT" ]; then
  echo "$ATT"
  echo "(failed attempts: the address after 'Connection to' is the one the app is using)"
else
  echo "no failed-connection lines (good, or the log buffer was cleared / app not started yet)"
fi
OVR=$($ADB logcat -d -s Unity 2>/dev/null | tr -d '\r' | grep "endpoint override" | tail -n 1)
[ -n "$OVR" ] && echo "override file in use: $OVR"

hr "ROS topics from the headset app and the video streamer"
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/ros2_ws/install/setup.bash 2>/dev/null
ros2 topic list 2>/dev/null | grep -E "q2r|vr/" || echo "none"
echo
echo "Firewall (run yourself if needed): sudo ufw status"
