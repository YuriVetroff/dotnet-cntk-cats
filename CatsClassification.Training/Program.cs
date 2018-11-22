using CatsClassification.Configuration;
using CatsClassification.DataInfrastructure;
using CatsClassification.Running;
using CatsClassification.Running.Responses;
using CNTK;
using System;
using System.IO;

using static System.Console;

namespace CatsClassification.Training
{
    internal static class CatsClassificationInitializer
    {
        private const int IMAGE_WIDTH = 224;
        private const int IMAGE_HEIGHT = 224;
        private const int IMAGE_DEPTH = 3;

        public static void Init(
            ClassificationConfig config,
            string configFile,
            string baseModelFile, string newModelFile,
            string trainImageFolder, string trainDatasetFile,
            string testImageFolder, string testDatasetFile)
        {
            CreateAndSaveModel(config, baseModelFile, newModelFile);
            CreateAndSaveDatasets(config, trainImageFolder, trainDatasetFile, testImageFolder, testDatasetFile);
            config.Save(configFile);
        }

        public static void CreateAndSaveModel(
            ClassificationConfig config, string baseModelFile, string newModelFile)
        {
            const string featureNodeName = "features";
            const string lastHiddenNodeName = "z.x";
            const string predictionNodeName = "prediction";
            var inputShape = new int[] { IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH };

            var device = DeviceDescriptor.CPUDevice;
            var model = CntkHelper.BuildTransferLearningModel(
                    Function.Load(baseModelFile, device),
                    featureNodeName,
                    predictionNodeName,
                    lastHiddenNodeName,
                    inputShape,
                    config.ClassCount,
                    device);

            model.Save(newModelFile);
        }
        public static void CreateAndSaveDatasets(
            ClassificationConfig config,
            string trainImageFolder, string trainDatasetFile,
            string testImageFolder, string testDatasetFile)
        {
            var datasetCreator = new ImageFolderDatasetCreator(
               config, IMAGE_WIDTH, IMAGE_HEIGHT);

            var dataFileCreator = new DataFileCreator();

            var trainDataset = datasetCreator.GetDataset(trainImageFolder);
            dataFileCreator.CreateDataFile(trainDataset, trainDatasetFile);

            var testDataset = datasetCreator.GetDataset(testImageFolder);
            dataFileCreator.CreateDataFile(testDataset, testDatasetFile);
        }
    }

    internal class Program
    {
        #region Constants

        private const string WORKING_DIRECTORY = "cats-classification";

        private const string TRAIN_IMAGE_FOLDER = "images/train";
        private const string TEST_IMAGE_FOLDER = "images/test";

        private const string TRAIN_DATASET_FILE = "train-dataset.txt";
        private const string TEST_DATASET_FILE = "test-dataset.txt";

        private const string BASE_MODEL_FILE = "ResNet18_ImageNet_CNTK.model";
        private const string NEW_MODEL_FILE = "cats-classifier.model";
        private const string TRAINED_MODEL_FILE = "cats-classifier-trained.model";

        private const string CONFIG_FILE = "config.txt";
        private const bool REQUIRE_INIT = true;

        #endregion

        #region Primary methods

        private static void Main(string[] args)
        {
            RunTraining(REQUIRE_INIT);
        }

        private static void RunTraining(bool requireInit)
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
                    () => Init());
            }

            var runner = RunnerFactory.GetRunnerForCats();
            runner.Mount(FinalizePath(NEW_MODEL_FILE));

            runner.TrainingIterationPerformed += TrainingIterationPerformed;
            runner.TrainingFinished += TrainingFinished;
            runner.TestingFinished += TestingFinished;

            runProcessAndWriteInConsole("Training", () => runner.Train(FinalizePath(TRAIN_DATASET_FILE)));
            runProcessAndWriteInConsole("Testing", () => runner.Test(FinalizePath(TEST_DATASET_FILE)));

            WriteLine("Press any key to exit...");
            ReadLine();
        }
        private static void Init()
        {
            var config = new ClassificationConfig(
                new string[]
                {
                    "Tiger",
                    "Leopard",
                    "Puma",
                    "Lynx",
                    "Lion"
                });

            CatsClassificationInitializer.Init(
                config,
                FinalizePath(CONFIG_FILE),
                FinalizePath(BASE_MODEL_FILE),
                FinalizePath(NEW_MODEL_FILE),
                FinalizePath(TRAIN_IMAGE_FOLDER),
                FinalizePath(TRAIN_DATASET_FILE),
                FinalizePath(TEST_IMAGE_FOLDER),
                FinalizePath(TEST_DATASET_FILE));
        }

        #endregion

        #region Training event handlers

        private static void TestingFinished(
            object sender, TestingResultResponse testingResultResponse) =>
            WriteLine(
                $"Testing finished. " +
                $"Correctly answered {testingResultResponse.Correct} of {testingResultResponse.Total}, " +
                $"accuracy = {testingResultResponse.Accuracy * 100}%");

        private static void TrainingFinished(
            object sender, TrainingResultResponse trainingResultResponse)
        {
            var finalPath = FinalizePath(TRAINED_MODEL_FILE);
            File.WriteAllBytes(finalPath, trainingResultResponse.NewModelData);
            WriteLine(
                $"Training finished. The model is saved at {finalPath}.");
        }

        private static void TrainingIterationPerformed(
            object sender, TrainingProgressResponse trainingProgressResponse) =>
            WriteLine(
                $"Minibatch: {trainingProgressResponse.MinibatchesSeen} " +
                $"CrossEntropyLoss = {trainingProgressResponse.Loss} " +
                $"EvaluationCriterion = {trainingProgressResponse.EvaluationCriterion}");

        #endregion

        #region Service methods

        private static string FinalizePath(string path) =>
            Path.Combine(WORKING_DIRECTORY, path);

        #endregion
    }
}
