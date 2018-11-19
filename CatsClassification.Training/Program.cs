using CNTK;
using System;
using System.IO;

using static System.Console;

namespace CatsClassification.Training
{
    internal class Program
    {
        private static string TRAIN_DATASET_FILE = "D:/train-dataset.txt";
        private static string TEST_DATASET_FILE = "D:/test-dataset.txt";
        private static string NEW_MODEL_FILE = "cats-classificator.model";
        private static string TRAINED_MODEL_FILE = "cats-classificator-trained.model";

        private static string _workingDirectory = "D:/cats-classification";
        private static Func<string, string> _finalizePath =
            (path) => Path.Combine(_workingDirectory, path);

        private static void Main(string[] args)
        {
            WriteLine("CATS CLASSIFICATION");
            WriteLine();

            var runner = new CatsClassificationRunner(_finalizePath(NEW_MODEL_FILE));

            runner.TrainingIterationPerformed += TrainingIterationPerformed;
            runner.TrainingFinished += TrainingFinished;
            runner.TestingFinished += TestingFinished;

            WriteLine("Training started...");
            runner.Train(_finalizePath(TRAIN_DATASET_FILE));
            WriteLine();

            WriteLine("Testing started...");
            runner.Test(_finalizePath(TEST_DATASET_FILE));
            WriteLine();

            WriteLine("Press any key to exit...");
            ReadLine();
        }

        private static void TestingFinished(
            object sender, TestingResult testingResult) =>
            WriteLine(
                $"Testing finished. " +
                $"Correctly answered {testingResult.Correct} of {testingResult.Total}," +
                $"accuracy = {testingResult.Accuracy * 100}%");

        private static void TrainingFinished(
            object sender, TrainingResult trainingResult)
        {
            var finalPath = _finalizePath(TRAINED_MODEL_FILE);
            File.WriteAllBytes(finalPath, trainingResult.NewModelData);
            WriteLine($"Training finished. The model is saved at {finalPath}.");
        }

        private static void TrainingIterationPerformed(
            object sender, TrainingProgress trainingProgress) =>
            WriteLine(
                $"Minibatch: {trainingProgress.MinibatchesSeen} " +
                $"CrossEntropyLoss = {trainingProgress.Loss} " +
                $"EvaluationCriterion = {trainingProgress.EvaluationCriterion}");

        private static void CreateAndSaveModel()
        {
            const string baseModelFile = "ResNet18_ImageNet_CNTK.model";
            const string featureNodeName = "features";
            const string lastHiddenNodeName = "z.x";
            const string predictionNodeName = "prediction";
            const int classCount = 3;
            int[] inputShape = new int[] { 224, 224, 3 };

            var device = DeviceDescriptor.CPUDevice;
            var model = CntkHelper.BuildTransferLearningModel(
                    Function.Load(_finalizePath(baseModelFile), device),
                    featureNodeName,
                    predictionNodeName,
                    lastHiddenNodeName,
                    inputShape,
                    classCount,
                    device);

            model.Save(_finalizePath(NEW_MODEL_FILE));
        }
    }
}
