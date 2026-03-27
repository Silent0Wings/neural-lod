// File: SampleRecord.cs
// Performance measurement for one (point, rotation, lodLevel) triple.
using System;
using UnityEngine;

[Serializable]
public class SampleRecord
{
    public int pointId;
    public Vector3 rotationAngles;
    public int lodLevel;
    public float meanGpuTimeMs;
    public float meanCpuTimeMs;
    public float meanFps;

    // scene complexity features added for fix 1
    public int   triangleCount;
    public int   visibleRendererCount;
    public float screenCoverage;
    public int   drawCallCount;
}
