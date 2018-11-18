using CNTK;
using System.IO;

namespace CatsClassification
{
    public class CatsClassificationRunner
    {
        private const int INPUT_IMAGE_SIDE = 224;
        private const int INPUT_IMAGE_CHANNEL_COUNT = 3;

        private const string FEATURE_NODE_NAME = "features";
        private const string LAST_HIDDEN_NODE_NAME = "z.x";

        private const string BASE_FOLDER = "Data";
        private const string BASE_MODEL_FILE = "/ResNet18_ImageNet_CNTK.model";

        private string FinalizePath(string finalPath) =>
            Path.Combine(BASE_FOLDER, finalPath);

        private int[] InputImageShape => new int[] { INPUT_IMAGE_SIDE, INPUT_IMAGE_SIDE, INPUT_IMAGE_CHANNEL_COUNT };
        private DeviceDescriptor Device => DeviceDescriptor.CPUDevice;
        
    }
}
