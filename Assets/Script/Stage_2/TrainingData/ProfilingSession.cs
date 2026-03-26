// File: ProfilingSession.cs
// Top-level container holding all data from one profiling run.
using System;
using System.Collections.Generic;

[Serializable]
public class ProfilingSession
{
    public string sessionId;
    public string sceneName;
    public List<LODObjectInfo> objects = new List<LODObjectInfo>();
    public List<GridPoint> grid = new List<GridPoint>();
    public List<SampleRecord> samples = new List<SampleRecord>();
    public List<LabelledSample> labels = new List<LabelledSample>();
}
