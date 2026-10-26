// QuestRosBridge.cs
// Publishes Quest 2 controller poses and inputs to ROS 2 (Unity ROS-TCP-Connector -> ROS-TCP-Endpoint)
// using the SAME topic names and message types as the Quest2ROS store app, and plays haptic requests.
//
// Calibration: hold A+B (right controller) or X+Y (left controller) together for ~1 s.
// Poses are then published relative to that controller's position and heading.
// NOTE: this is my own calibration design; the store app's exact frame definition was not verified.
//
// Written without a Unity compiler. If a message class or field name differs, open the generated files
// in Assets/RosMessages/ and adjust the names below.
using System;
using TMPro;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Geometry;
using RosMessageTypes.Quest2ros;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using UnityEngine;
using UnityEngine.XR;

[DefaultExecutionOrder(-100)]
public class QuestRosBridge : MonoBehaviour
{
    [Header("Topics (same names as the Quest2ROS store app)")]
    public string rightPoseTopic = "q2r_right_hand_pose";
    public string leftPoseTopic = "q2r_left_hand_pose";
    public string rightInputsTopic = "q2r_right_hand_inputs";
    public string leftInputsTopic = "q2r_left_hand_inputs";
    public string rightHapticTopic = "q2r_right_hand_haptic_feedback";
    public string leftHapticTopic = "q2r_left_hand_haptic_feedback";
    public string frameId = "q2r_world";

    [Header("Extra (NOT in the store app): thumbstick click as std_msgs/Bool")]
    public bool publishThumbstickClick = true;
    public string rightThumbClickTopic = "q2r_right_hand_thumbstick_click";
    public string leftThumbClickTopic = "q2r_left_hand_thumbstick_click";

    [Header("Runtime endpoint override (no rebuild needed)")]
    [Tooltip("If <persistentDataPath>/ros_ip.txt exists (adb push it to /sdcard/Android/data/<package>/files/), " +
             "its content 'ip' or 'ip:port' replaces the address set in ROS Settings.")]
    public string overrideFileName = "ros_ip.txt";

    [Header("Calibration")]
    public float calibrationHoldSeconds = 1.0f;
    [Tooltip("true: frame = controller position + heading only (gravity stays up). false: full controller orientation.")]
    public bool yawOnlyCalibration = true;

    [Header("Optional: re-center the video screen in front of the head on calibration")]
    public Transform screen;      // parent object of the video quad
    public Transform head;        // the Main Camera under XR Origin
    public float screenDistance = 1.6f;

    [Header("Optional HUD (world-space TextMeshPro)")]
    public TextMeshPro hud;
    public AmbfVideoQuad video;

    class HandState
    {
        public XRNode node;
        public string poseTopic, inputsTopic, thumbClickTopic;
        public float calibHold;
        public bool calibLatched;
        public volatile float pendingHaptic = -1f;
        public bool tracked;
        public int sent;
    }

    ROSConnection ros;
    readonly HandState right = new HandState { node = XRNode.RightHand };
    readonly HandState left = new HandState { node = XRNode.LeftHand };

    Vector3 refPos = Vector3.zero;
    Quaternion refRot = Quaternion.identity;
    bool calibrated;
    float nextHud;

    void Awake()
    {
        ROSConnection conn = ROSConnection.GetOrCreateInstance();
        try
        {
            string path = System.IO.Path.Combine(Application.persistentDataPath, overrideFileName);
            if (System.IO.File.Exists(path))
            {
                string[] p = System.IO.File.ReadAllText(path).Trim().Split(':');
                if (p[0].Length > 0) conn.RosIPAddress = p[0];
                if (p.Length > 1 && int.TryParse(p[1], out int port)) conn.RosPort = port;
                Debug.Log("QuestRosBridge: endpoint override from " + path + " -> " + conn.RosIPAddress + ":" + conn.RosPort);
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("QuestRosBridge: could not read " + overrideFileName + ": " + e.Message);
        }
    }

    void Start()
    {
        right.poseTopic = rightPoseTopic; right.inputsTopic = rightInputsTopic; right.thumbClickTopic = rightThumbClickTopic;
        left.poseTopic = leftPoseTopic; left.inputsTopic = leftInputsTopic; left.thumbClickTopic = leftThumbClickTopic;

        ros = ROSConnection.GetOrCreateInstance();
        foreach (var h in new[] { right, left })
        {
            ros.RegisterPublisher<PoseStampedMsg>(h.poseTopic);
            ros.RegisterPublisher<OVR2ROSInputsMsg>(h.inputsTopic);
            if (publishThumbstickClick) ros.RegisterPublisher<BoolMsg>(h.thumbClickTopic);
        }
        // Haptic requests arrive on a network thread: only store them here, play them in Update().
        ros.Subscribe<OVR2ROSHapticFeedbackMsg>(rightHapticTopic, m => right.pendingHaptic = m.amplitude);
        ros.Subscribe<OVR2ROSHapticFeedbackMsg>(leftHapticTopic, m => left.pendingHaptic = m.amplitude);
    }

    void Update()
    {
        ServiceHand(right);
        ServiceHand(left);
        UpdateHud();
    }

    void ServiceHand(HandState h)
    {
        InputDevice dev = InputDevices.GetDeviceAtXRNode(h.node);
        if (!dev.isValid) { h.tracked = false; return; }

        if (h.pendingHaptic >= 0f)
        {
            Pulse(dev, h.pendingHaptic, 0.1f);   // the XR API has no frequency control; amplitude only
            h.pendingHaptic = -1f;
        }

        dev.TryGetFeatureValue(CommonUsages.isTracked, out bool tracked);
        bool hasPos = dev.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 pos);
        bool hasRot = dev.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rot);
        dev.TryGetFeatureValue(CommonUsages.primaryButton, out bool primary);     // A (right) / X (left)
        dev.TryGetFeatureValue(CommonUsages.secondaryButton, out bool secondary); // B (right) / Y (left)
        dev.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 stick);
        dev.TryGetFeatureValue(CommonUsages.primary2DAxisClick, out bool stickClick);
        dev.TryGetFeatureValue(CommonUsages.trigger, out float trigger);
        dev.TryGetFeatureValue(CommonUsages.grip, out float grip);
        h.tracked = tracked;

        // --- calibration gesture: both face buttons held together ---
        if (hasPos && hasRot && primary && secondary)
        {
            h.calibHold += Time.unscaledDeltaTime;
            if (!h.calibLatched && h.calibHold >= calibrationHoldSeconds)
            {
                Calibrate(pos, rot);
                h.calibLatched = true;
                Pulse(dev, 0.8f, 0.3f);
            }
        }
        else
        {
            h.calibHold = 0f;
            h.calibLatched = false;
        }

        // --- pose ---
        if (tracked && hasPos && hasRot)
        {
            Vector3 relPos = Quaternion.Inverse(refRot) * (pos - refPos);
            Quaternion relRot = Quaternion.Inverse(refRot) * rot;
            var pose = new PoseStampedMsg
            {
                header = new HeaderMsg { stamp = Now(), frame_id = frameId },
                pose = new PoseMsg
                {
                    position = relPos.To<FLU>(),       // Unity (x right, y up, z fwd) -> ROS (x fwd, y left, z up)
                    orientation = relRot.To<FLU>()
                }
            };
            ros.Publish(h.poseTopic, pose);
            h.sent++;
        }

        // --- inputs (mapping of button_upper/lower is my assumption: B/Y = upper, A/X = lower) ---
        var inputs = new OVR2ROSInputsMsg
        {
            button_upper = secondary,
            button_lower = primary,
            thumb_stick_horizontal = stick.x,
            thumb_stick_vertical = stick.y,
            press_index = trigger,
            press_middle = grip
        };
        ros.Publish(h.inputsTopic, inputs);
        if (publishThumbstickClick) ros.Publish(h.thumbClickTopic, new BoolMsg { data = stickClick });
    }

    void Calibrate(Vector3 pos, Quaternion rot)
    {
        refPos = pos;
        if (yawOnlyCalibration)
        {
            Vector3 fwd = Vector3.ProjectOnPlane(rot * Vector3.forward, Vector3.up);
            if (fwd.sqrMagnitude > 1e-4f)
                refRot = Quaternion.LookRotation(fwd.normalized, Vector3.up);
        }
        else
        {
            refRot = rot;
        }
        calibrated = true;

        if (screen != null && head != null)
        {
            Vector3 f = Vector3.ProjectOnPlane(head.forward, Vector3.up);
            if (f.sqrMagnitude > 1e-4f)
            {
                f.Normalize();
                screen.position = head.position + f * screenDistance;
                screen.rotation = Quaternion.LookRotation(f, Vector3.up);
            }
        }
    }

    static void Pulse(InputDevice dev, float amplitude, float seconds)
    {
        if (dev.TryGetHapticCapabilities(out HapticCapabilities caps) && caps.supportsImpulse)
            dev.SendHapticImpulse(0u, Mathf.Clamp01(amplitude), seconds);
    }

    // Quest wall clock as a ROS time stamp (the store app leaves stamps at 0).
    // Only as accurate as the headset's clock; use it for ordering/rates, not absolute latency.
    static TimeMsg Now()
    {
        double s = (DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds;
        int sec = (int)Math.Floor(s);
        return new TimeMsg { sec = sec, nanosec = (uint)((s - sec) * 1e9) };
    }

    void UpdateHud()
    {
        if (hud == null || Time.unscaledTime < nextHud) return;
        nextHud = Time.unscaledTime + 0.25f;
        string v = video == null ? "" :
            $"\nvideo: {video.Frames} frames | age {video.AgeMs:0} ms | last {video.SecondsSinceFrame:0.0} s ago";
        hud.text =
            $"ROS endpoint: {ros.RosIPAddress}:{ros.RosPort}\n" +   // if this line fails to compile, delete it
            $"calibrated: {(calibrated ? "yes" : "NO  (hold A+B or X+Y)")}\n" +
            $"R tracked: {right.tracked}  sent: {right.sent}\n" +
            $"L tracked: {left.tracked}  sent: {left.sent}" + v;
    }
}
