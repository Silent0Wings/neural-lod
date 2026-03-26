// File: LabelledSample.cs
// Links a (pointId, rotation) viewpoint to its oracle-optimal threshold vector.
using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class LabelledSample
{
    public int pointId;
    public Vector3 rotationAngles;
    public List<float> optimalThresholds = new List<float>();
}
