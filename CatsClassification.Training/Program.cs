using CNTK;
using System;
using System.Collections.Generic;
using System.IO;

namespace CatsClassification.Training
{
    internal class Program
    {
        private static string TRAIN_DATASET_FILE = "D:/train-dataset.txt";
        private static string TEST_DATASET_FILE = "D:/test-dataset.txt";
        private static string NEW_MODEL_FILE = "cats-classificator.model";

        private static string _workingDirectory = "D:/cats-classification";
        private static Func<string, string> _finalizePath =
            (path) => Path.Combine(_workingDirectory, path);
        private static DeviceDescriptor _device = DeviceDescriptor.CPUDevice;

        private static void Main(string[] args)
        {
            Console.WriteLine("Cats classification");

            var runner = new CatsClassificationRunner(_finalizePath(NEW_MODEL_FILE), _device);
            runner.Train(_finalizePath(TRAIN_DATASET_FILE));
            runner.Test(_finalizePath(TEST_DATASET_FILE));
            Console.ReadLine();
        }

        private static void CreateAndSaveModel()
        {
            const string baseModelFile = "ResNet18_ImageNet_CNTK.model";
            const string featureNodeName = "features";
            const string lastHiddenNodeName = "z.x";
            const string predictionNodeName = "prediction";
            const int classCount = 3;
            int[] inputShape = new int[] { 224, 224, 3 };

            var model = CntkHelper.BuildTransferLearningModel(
                Function.Load(_finalizePath(baseModelFile), _device),
                featureNodeName,
                predictionNodeName,
                lastHiddenNodeName,
                inputShape,
                classCount,
                _device);

            model.Save(_finalizePath(NEW_MODEL_FILE));
        }
    }
}
