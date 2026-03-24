using UnityEngine;
using Unity.InferenceEngine;

public class ModelLoadTest : MonoBehaviour
{
    public ModelAsset onnxAsset;

    void Start()
    {
        var model = ModelLoader.Load(onnxAsset);
        var worker = new Worker(model, BackendType.CPU);

        Debug.Log($"[ModelLoadTest] Model loaded OK");
        Debug.Log($"[ModelLoadTest] Input  name: {model.inputs[0].name}");
        Debug.Log($"[ModelLoadTest] Output name: {model.outputs[0].name}");

        worker.Dispose();
    }
    private const int EXPECTED_FEATURE_COUNT = 20;

    void Start()
    {
        if (onnxAsset == null)
        {
            Debug.LogError("[ModelLoadTest] onnxAsset is not assigned.");
            return;
        }

        var model  = ModelLoader.Load(onnxAsset);
        var worker = new Worker(model, BackendType.CPU);

        Debug.Log("[ModelLoadTest] Model loaded OK");
        Debug.Log($"[ModelLoadTest] Input  name: {model.inputs[0].name}");
        Debug.Log($"[ModelLoadTest] Output name: {model.outputs[0].name}");

        // check input tensor rank
        var inputShape = model.inputs[0].shape;
        int rank = inputShape.rank;
        Debug.Log($"[ModelLoadTest] Input shape rank: {rank}");

        if (rank < 2)
        {
            Debug.LogError($"[ModelLoadTest] CONTRA-05: Input rank {rank} is less than 2. Expected rank 2 shape [1, {EXPECTED_FEATURE_COUNT}].");
        }
        else
        {
            // dim index 1 is the feature dimension for shape [batch, features]
            int featureDim = inputShape[1].value;
            if (featureDim != EXPECTED_FEATURE_COUNT)
                Debug.LogError($"[ModelLoadTest] CONTRA-05: Input feature dim = {featureDim}, " +
                            $"expected {EXPECTED_FEATURE_COUNT}. Wrong model version deployed.");
            else
                Debug.Log($"[ModelLoadTest] Input shape OK — [{inputShape[0].value}, {featureDim}]");
        }

        // check output is scalar or [1,1]
        var outputShape = model.outputs[0].shape;
        if (outputShape.rank >= 2 && outputShape[1].value != 1)
            Debug.LogWarning($"[ModelLoadTest] Output feature dim = {outputShape[1].value}, expected 1. Check model head.");
        else
            Debug.Log($"[ModelLoadTest] Output shape OK.");

        worker.Dispose();
    }
    
}