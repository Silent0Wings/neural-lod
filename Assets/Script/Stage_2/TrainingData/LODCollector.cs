// File: LODCollector.cs
// Scans the active scene for LODGroup components.
// Builds LODObjectInfo entries with stable reference IDs and threshold lists.
// Uses Unity 6 API: Object.FindObjectsByType instead of deprecated FindObjectsOfType.
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections.Generic;

public class LODCollector : MonoBehaviour
{
    // collected results accessible after ScanScene
    [HideInInspector] public List<LODObjectInfo> collectedObjects = new List<LODObjectInfo>();

    // all discovered LODGroups for downstream use
    [HideInInspector] public LODGroup[] lodGroups;

    /// call once before profiling begins
    public void ScanScene()
    {
        collectedObjects.Clear();

        // Unity 6 modern API
        lodGroups = Object.FindObjectsByType<LODGroup>(FindObjectsSortMode.None);

        string sceneName = SceneManager.GetActiveScene().name;

        foreach (LODGroup group in lodGroups)
        {
            LODObjectInfo info = new LODObjectInfo();

            // stable id from scene name plus instance id
            info.referenceId = sceneName + ":" + group.gameObject.GetInstanceID();

            LOD[] lods = group.GetLODs();
            info.rendererCount = 0;

            for (int i = 0; i < lods.Length; i++)
            {
                info.thresholds.Add(lods[i].screenRelativeTransitionHeight);
                if (lods[i].renderers != null)
                    info.rendererCount += lods[i].renderers.Length;
            }

            collectedObjects.Add(info);
        }

        Debug.Log($"[LODCollector] Found {collectedObjects.Count} LODGroups in scene '{sceneName}'.");
    }
}
