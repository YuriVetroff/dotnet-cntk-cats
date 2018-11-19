using CNTK;
using System;
using System.Collections.Generic;
using System.IO;

using static System.Console;

namespace CatsClassification.Training
{
    internal static class CatsClassificationInitializer
    {
        private const int IMAGE_WIDTH = 224;
        private const int IMAGE_HEIGHT = 224;
        private const int IMAGE_DEPTH = 3;
        private const int CLASS_COUNT = 3;

        public static void Init(
            string baseModelFile, string newModelFile,
            string trainImageFolder, string trainDatasetFile,
            string testImageFolder, string testDatasetFile)
        {
            CreateAndSaveModel(baseModelFile, newModelFile);
            CreateAndSaveDatasets(trainImageFolder, trainDatasetFile, testImageFolder, testDatasetFile);
        }

        private static void CreateAndSaveModel(string baseModelFile, string newModelFile)
        {
            const string featureNodeName = "features";
            const string lastHiddenNodeName = "z.x";
            const string predictionNodeName = "prediction";
            int[] inputShape = new int[] { IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH };

            var device = DeviceDescriptor.CPUDevice;
            var model = CntkHelper.BuildTransferLearningModel(
                    Function.Load(baseModelFile, device),
                    featureNodeName,
                    predictionNodeName,
                    lastHiddenNodeName,
                    inputShape,
                    CLASS_COUNT,
                    device);

            model.Save(newModelFile);
        }
        private static void CreateAndSaveDatasets(
            string trainImageFolder, string trainDatasetFile,
            string testImageFolder, string testDatasetFile)
        {
            var datasetCreator = new ImageFolderDatasetCreator(
                new Dictionary<string, int>
                {
                    {  "Tiger", 0 },
                    {  "Leopard", 1 },
                    {  "Puma", 2 }
                }, CLASS_COUNT, IMAGE_WIDTH, IMAGE_HEIGHT);

            var dataFileCreator = new DataFileCreator();

            var trainDataset = datasetCreator.GetDataset(trainImageFolder);
            dataFileCreator.CreateDataFile(trainDataset, trainDatasetFile);

            var testDataset = datasetCreator.GetDataset(testImageFolder);
            dataFileCreator.CreateDataFile(testDataset, testDatasetFile);
        }
    }

    internal class Program
    {
        private static string WORKING_DIRECTORY = "cats-classification";

        private static string TRAIN_DATASET_FILE = "train-dataset.txt";
        private static string TEST_DATASET_FILE = "test-dataset.txt";

        private static string BASE_MODEL_FILE = "ResNet18_ImageNet_CNTK.model";
        private static string NEW_MODEL_FILE = "cats-classificator.model";
        private static string TRAINED_MODEL_FILE = "cats-classificator-trained.model";

        private static string FinalizePath(string path) =>
            Path.Combine(WORKING_DIRECTORY, path);

        private static void Main(string[] args)
        {
            const bool requireInit = false;
            RunTraining(requireInit);
        }

        private static void RunTraining(bool requireInit = false)
        {
            WriteLine("CATS CLASSIFICATION");
            WriteLine();

            Action<string, Action> runProcessAndWriteInConsole =
                (processName, action) =>
                {
                    WriteLine($"{processName}...");
                    action();
                    WriteLine();
                };

            if (requireInit)
            {
                runProcessAndWriteInConsole("Initialization",
                    () => CatsClassificationInitializer.Init(
                        FinalizePath(BASE_MODEL_FILE),
                        FinalizePath(NEW_MODEL_FILE),
                        FinalizePath("D:/Datasets/Animals-cats/Train"),
                        FinalizePath(TRAIN_DATASET_FILE),
                        FinalizePath("D:/Datasets/Animals-cats/Test"),
                        FinalizePath(TEST_DATASET_FILE)));
            }

            var runner = new CatsClassificationRunner(FinalizePath(NEW_MODEL_FILE));

            runner.TrainingIterationPerformed += TrainingIterationPerformed;
            runner.TrainingFinished += TrainingFinished;
            runner.TestingFinished += TestingFinished;

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
    }
}
