using UnityEngine;
using Unity.InferenceEngine;

public class ModelLoadTest : MonoBehaviour
{
    public ModelAsset onnxAsset;
    private const int EXPECTED_FEATURE_COUNT = 20;

    void Start()
    {
        if (onnxAsset == null)
        {
            Debug.LogError("[ModelLoadTest] onnxAsset is not assigned.");
            return;
        }

        var model = ModelLoader.Load(onnxAsset);
        var worker = new Worker(model, BackendType.CPU);

        Debug.Log("[ModelLoadTest] Model loaded OK");
        Debug.Log($"[ModelLoadTest] Input  name: {model.inputs[0].name}");
        Debug.Log($"[ModelLoadTest] Output name: {model.outputs[0].name}");

        var inputShape = model.inputs[0].shape;
        int rank = inputShape.rank;
        string shapeStr = inputShape.ToString();

        Debug.Log($"[ModelLoadTest] Input rank: {rank}  shape: {shapeStr}");

        if (rank < 2)
        {
            Debug.LogError($"[ModelLoadTest] CONTRA-05: Input rank {rank} < 2. " +
                           $"Expected [1, {EXPECTED_FEATURE_COUNT}].");
        }
        else
        {
            if (!shapeStr.Contains(EXPECTED_FEATURE_COUNT.ToString()))
                Debug.LogError($"[ModelLoadTest] CONTRA-05: Feature dim {EXPECTED_FEATURE_COUNT} " +
                               $"not found in shape {shapeStr}. Wrong model deployed.");
            else
                Debug.Log($"[ModelLoadTest] Input shape OK — {shapeStr}");
        }

        Debug.Log($"[ModelLoadTest] Output name: {model.outputs[0].name}");

        worker.Dispose();
    }
}