// File: LODObjectInfo.cs
// Stores reference ID and per-LOD transition thresholds for one LODGroup.
using System;
using System.Collections.Generic;

[Serializable]
public class LODObjectInfo
{
    public string referenceId;
    public List<float> thresholds = new List<float>();
    public int rendererCount;
}
