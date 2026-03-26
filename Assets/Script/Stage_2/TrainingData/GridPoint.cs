// File: GridPoint.cs
// Represents one cell center in the 3D sampling grid with its volume bounds.
using System;
using UnityEngine;

[Serializable]
public class GridPoint
{
    public int pointId;
    public Vector3 coordinates;
    public Vector3 volumeBoundsMin;
    public Vector3 volumeBoundsMax;
}
