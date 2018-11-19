using CNTK;
using System;
using System.IO;

using static System.Console;

namespace CatsClassification.Training
{
    internal class Program
    {
        private static string TRAIN_DATASET_FILE = "train-dataset.txt";
        private static string TEST_DATASET_FILE = "test-dataset.txt";
        private static string NEW_MODEL_FILE = "cats-classificator.model";
        private static string TRAINED_MODEL_FILE = "cats-classificator-trained.model";

        private static string WORKING_DIRECTORY = "D:/cats-classification";

        private static string FinalizePath(string path) =>
            Path.Combine(WORKING_DIRECTORY, path);

        private static void Main(string[] args)
        {
            WriteLine("CATS CLASSIFICATION");
            WriteLine();

            var runner = new CatsClassificationRunner(FinalizePath(NEW_MODEL_FILE));

            runner.TrainingIterationPerformed += TrainingIterationPerformed;
            runner.TrainingFinished += TrainingFinished;
            runner.TestingFinished += TestingFinished;

            Action<string, Action> runProcessAndWriteInConsole =
                (processName, action) =>
                {
                    WriteLine($"{processName} started...");
                    action();
                    WriteLine();
                };

            runProcessAndWriteInConsole("Training", () => runner.Train(FinalizePath(TRAIN_DATASET_FILE)));
            runProcessAndWriteInConsole("Testing", () => runner.Test(FinalizePath(TEST_DATASET_FILE)));

            WriteLine("Press any key to exit...");
            ReadLine();
        }

        private static void TestingFinished(
            object sender, TestingResult testingResult) =>
            WriteLine(
                $"Testing finished. " +
                $"Correctly answered {testingResult.Correct} of {testingResult.Total}, " +
                $"accuracy = {testingResult.Accuracy * 100}%");

        private static void TrainingFinished(
            object sender, TrainingResult trainingResult)
        {
            var finalPath = FinalizePath(TRAINED_MODEL_FILE);
            File.WriteAllBytes(finalPath, trainingResult.NewModelData);
            WriteLine(
                $"Training finished. The model is saved at {finalPath}.");
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
                    Function.Load(FinalizePath(baseModelFile), device),
                    featureNodeName,
                    predictionNodeName,
                    lastHiddenNodeName,
                    inputShape,
                    classCount,
                    device);

            model.Save(FinalizePath(NEW_MODEL_FILE));
        }
    }
}
