using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace CatsClassification
{
    public class TransferLearning
    {
        protected const string FEATURE_STREAM_NAME = "features";
        protected const string LABEL_STREAM_NAME = "labels";

        protected const string DATA_FILE = "dataset.txt";

        public static string CurrentFolder = "D:/";

        public static string ExampleImageFolder
        {
            get => TestCommon.TestDataDirPrefix;
            set => TestCommon.TestDataDirPrefix = value;
        }

        public static string BaseResnetModelFile = TestCommon.TestDataDirPrefix + "/ResNet18_ImageNet_CNTK.model";

        private static string featureNodeName = "features";
        private static string lastHiddenNodeName = "z.x";
        private static int[] imageDims = new int[] { 224, 224, 3 };

        private static string _baseDataFolder = Path.Combine(ExampleImageFolder, "Datasets/Animals-cats");
        private static string _trainFolderPrefix = "Train";
        private static string _testFolderPrefix = "Test";
        private static ImageFolderDatasetCreator _datasetCreator =
            new ImageFolderDatasetCreator(new Dictionary<string, int>
            {
                { "Tiger", 0 },
                { "Leopard", 1 },
                { "Puma", 2 },
            }, 3, imageDims[0], imageDims[1]);

        public static void Train(DeviceDescriptor device, bool forceRetrain = true)
        {
            int numClasses = 3;
            string modelFile = Path.Combine(CurrentFolder, "AnimalsTransferLearning.model");

            var trainFolder = Path.Combine(_baseDataFolder, _trainFolderPrefix);
            var dataset = _datasetCreator.GetDataset(trainFolder);

            // prepare the transfer model
            string predictionNodeName = "prediction";
            Function model = CntkHelper.GetModel(
                Path.Combine(ExampleImageFolder, BaseResnetModelFile),
                featureNodeName,
                predictionNodeName,
                lastHiddenNodeName,
                imageDims,
                numClasses,
                device);
            
            // prepare for training
            int maxMinibatches = 5;
            float learningRate = 0.2F;
            float momentum = 0.9F;
            float l2regularization = 0.1F;

            AdditionalLearningOptions additionalLearningOptions =
                new AdditionalLearningOptions() { l2RegularizationWeight = l2regularization };
            IList<Learner> parameterLearners = new List<Learner>() {
                    Learner.MomentumSGDLearner(model.Parameters(),
                    new TrainingParameterScheduleDouble(learningRate, 0),
                    new TrainingParameterScheduleDouble(momentum, 0),
                    true,
                    additionalLearningOptions)};

            var input = model.Arguments[0];
            var output = Variable.InputVariable(new int[] { numClasses }, DataType.Float);
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(model, output);
            var predictionError = CNTKLib.ClassificationError(model, output);
            var trainer = Trainer.CreateTrainer(model, trainingLoss, predictionError, parameterLearners);

            var dataSource = InitDataSource(dataset, DATA_FILE);

            var currentMinibatch = 0;
            uint minibatchSize = 15;
            while (true)
            {
                var minibatchData = dataSource.MinibatchSource.GetNextMinibatch(minibatchSize, device);
                var arguments = new Dictionary<Variable, MinibatchData>
                    {
                        { input, minibatchData[dataSource.FeatureStreamInfo] },
                        { output, minibatchData[dataSource.LabelStreamInfo] }
                    };
                trainer.TrainMinibatch(arguments, device);
                CntkHelper.PrintTrainingProgress(trainer, currentMinibatch, 1);
                currentMinibatch++;
                if (currentMinibatch >= maxMinibatches)
                {
                    break;
                }
            }
            
            model.Save(modelFile);
            
            Test(modelFile, Path.Combine(_baseDataFolder, "Test"), imageDims, numClasses, device);

            Console.ReadLine();
        }

        private static void Test(string modelFile, string testDataFolder,
            int[] imageDims, int numClasses, DeviceDescriptor device)
        {
            var testFolder = Path.Combine(_baseDataFolder, _testFolderPrefix);
            var dataset = _datasetCreator.GetDataset(testFolder);

            Function model = Function.Load(modelFile, device);
            var input = model.Arguments[0];
            var output = model.Output;

            int mistakes = 0, total = 0;

            var dataSource = InitDataSource(dataset, DATA_FILE);
            var currentMinibatch = 0;
            uint minibatchSize = 1;
            while (true)
            {
                var minibatchData = dataSource.MinibatchSource.GetNextMinibatch(minibatchSize, device);
                var inputDataMap = new Dictionary<Variable, Value>() { { input, minibatchData[dataSource.FeatureStreamInfo].data } };
                var outputDataMap = new Dictionary<Variable, Value>() { { output, null } };

                model.Evaluate(inputDataMap, outputDataMap, device);
                var outputVal = outputDataMap[output];
                var actual = outputVal.GetDenseData<float>(output);
                var labelBatch = minibatchData[dataSource.LabelStreamInfo].data;
                var expected = labelBatch.GetDenseData<float>(model.Output);

                var actualLabels = actual.Select((IList<float> l) => l.IndexOf(l.Max())).ToList();
                var expectedLabels = expected.Select((IList<float> l) => l.IndexOf(l.Max())).ToList();

                int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();
                mistakes += misMatches;
                total += actualLabels.Count();

                currentMinibatch++;
                if (minibatchData.Values.Any(x => x.sweepEnd))
                {
                    break;
                }
            }
            Console.WriteLine($"Validating Model: Total Samples = {total}, Misclassify Count = {mistakes}");
            
            var error = 1.0 * mistakes / dataset.Items.Count();
        }

        private static Dictionary<string, int> LoadMapFile(string mapFile)
        {
            Dictionary<string, int> imageFileToLabel = new Dictionary<string, int>();
            string line;

            if (File.Exists(mapFile))
            {
                StreamReader file = null;
                try
                {
                    file = new StreamReader(mapFile);
                    while ((line = file.ReadLine()) != null)
                    {
                        int spaceIndex = line.IndexOfAny(new char[] { ' ', '\t' });
                        string filePath = line.Substring(0, spaceIndex);
                        int label = int.Parse(line.Substring(spaceIndex).Trim());
                        imageFileToLabel.Add(filePath, label);
                    }
                }
                finally
                {
                    if (file != null)
                        file.Close();
                }
            }
            return imageFileToLabel;
        }

        private static CntkDataSource InitDataSource(Dataset dataset, string filename)
        {
            var inputLength = dataset.Items.First().Input.Length;
            var outputLength = dataset.Items.First().Output.Length;
            var streamConfigs = new StreamConfiguration[]
            {
                new StreamConfiguration(FEATURE_STREAM_NAME, inputLength),
                new StreamConfiguration(LABEL_STREAM_NAME, outputLength)
            };

            var creator = new DataFileCreator();
            creator.CreateDataFile(dataset, filename);

            return new CntkDataSource(MinibatchSource.TextFormatMinibatchSource(
                filename,
                streamConfigs,
                MinibatchSource.InfinitelyRepeat), FEATURE_STREAM_NAME, LABEL_STREAM_NAME);
        }
    }
}
