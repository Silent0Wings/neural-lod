using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneController : MonoBehaviour
{
    /// <summary>
    /// Loads a scene by its name. Make sure the scene is added in the Build Settings.
    /// </summary>
    public void LoadScene(string sceneName)
    {
        SceneManager.LoadScene(sceneName);
    }

    /// <summary>
    /// Quits the program. If playing in the Unity Editor, it will stop playing.
    /// </summary>
    public void QuitProgram()
    {
        Debug.Log("Quitting program...");
        
        #if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
        #else
        Application.Quit();
        #endif
    }
}
