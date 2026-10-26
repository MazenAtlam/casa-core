// AmbfVideoQuad.cs
// Subscribes to a sensor_msgs/CompressedImage (JPEG) topic through ROS-TCP-Connector and shows
// the newest frame on a quad. Frames are decoded on the main thread; only the newest one is kept.
using System;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

public class AmbfVideoQuad : MonoBehaviour
{
    public string topic = "/vr/left/image/compressed";
    [Tooltip("Renderer of the Quad (material: Unlit/Texture)")]
    public Renderer quad;

    public int Frames { get; private set; }
    /// <summary>Smoothed age of the frame when displayed, from its header stamp. Needs PC and Quest clocks in sync (NTP).</summary>
    public float AgeMs { get; private set; }
    public float SecondsSinceFrame => Frames == 0 ? -1f : Time.realtimeSinceStartup - lastFrame;

    Texture2D tex;
    readonly object gate = new object();
    byte[] pending;
    double pendingStamp;
    float lastFrame;
    static readonly DateTime Epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    void Start()
    {
        tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
        // The quad must have exactly ONE material (Unlit/Texture). With two slots Unity draws both over the
        // same triangles and the last one (empty = white) hides the video. Set the texture on all slots anyway.
        if (quad.sharedMaterials.Length != 1)
            Debug.LogWarning("AmbfVideoQuad: the Quad has " + quad.sharedMaterials.Length +
                             " materials; set Mesh Renderer > Materials > Size to 1 (Unlit/Texture).");
        foreach (var m in quad.materials) m.mainTexture = tex;
        ROSConnection.GetOrCreateInstance().Subscribe<CompressedImageMsg>(topic, OnImage);
    }

    // May run on a network thread: just hand the bytes over.
    void OnImage(CompressedImageMsg m)
    {
        lock (gate)
        {
            pending = m.data;
            pendingStamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
        }
    }

    void Update()
    {
        byte[] data;
        double stamp;
        lock (gate)
        {
            data = pending;
            stamp = pendingStamp;
            pending = null;
        }
        if (data == null) return;

        if (tex.LoadImage(data, false))   // resizes the texture to the JPEG size
        {
            Frames++;
            lastFrame = Time.realtimeSinceStartup;
            double now = (DateTime.UtcNow - Epoch).TotalSeconds;
            AgeMs = Mathf.Lerp(AgeMs, (float)((now - stamp) * 1000.0), 0.1f);
        }
    }
}
