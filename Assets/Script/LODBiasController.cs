using UnityEngine;

/// <summary>
/// LODBiasController
/// Set lodBias via the Inspector. Changes apply on Play.
/// Attach to any GameObject in the scene.
/// </summary>
public class LODBiasController : MonoBehaviour
{
    [Range(0.1f, 100.0f)]
    public float lodBias = 1.0f;

    void Awake()
    {
        UpdateBias(lodBias);
        LogLODBias();
    }

    public void UpdateBias(float f)
    {
        QualitySettings.lodBias = f;
        lodBias = f;
    }

    void LogLODBias()
    {
        float baseline = QualitySettings.lodBias;
        string qualityLevel = QualitySettings.names[QualitySettings.GetQualityLevel()];
        Debug.Log($"[LODBaselineLogger] Quality Level: {qualityLevel} | LOD Bias: {baseline}");
    }
}
