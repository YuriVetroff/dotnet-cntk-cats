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

        private static string _trainDataFile = "D:/train-dataset.txt";
        private static string _testDataFile = "D:/test-dataset.txt";

        public static string _baseFolder = "D:/";
        public static string _baseModelFile = _baseFolder + "/ResNet18_ImageNet_CNTK.model";

        private static string _featureNodeName = "features";
        private static string _lastHiddenNodeName = "z.x";
        private static string _predictionNodeName = "prediction";
        private static int[] _inputShape = new int[] { 224, 224, 3 };
        private static int[] _outputShape = new int[] { _classCount };

        private static string _baseDataFolder = Path.Combine(_baseFolder, "Datasets/Animals-cats");
        private static string _trainFolderPrefix = "Train";
        private static string _testFolderPrefix = "Test";

        private static DeviceDescriptor _device = DeviceDescriptor.CPUDevice;

        private static IReadOnlyDictionary<string, int> _classMap = new Dictionary<string, int>
        {
            { "Tiger", 0 },
            { "Leopard", 1 },
            { "Puma", 2 },
        };
        private static int _classCount = _classMap.Count;
        private static ImageFolderDatasetCreator _datasetCreator =
            new ImageFolderDatasetCreator(_classMap, _classCount, _inputShape[0], _inputShape[1]);

        static TransferLearning() { }

        public static void Train()
        {
            var trainFolder = Path.Combine(_baseDataFolder, _trainFolderPrefix);
            var dataset = _datasetCreator.GetDataset(trainFolder);
            
            Function model = CntkHelper.BuildTransferLearningModel(
                Function.Load(Path.Combine(_baseFolder, _baseModelFile), _device),
                _featureNodeName,
                _predictionNodeName,
                _lastHiddenNodeName,
                _inputShape,
                _classCount,
                _device);
            
            // prepare for training
            int maxMinibatches = 5;
            float learningRate = 0.2F;
            float momentum = 0.9F;
            float l2regularization = 0.1F;

            var input = model.Arguments[0];
            var output = Variable.InputVariable(new int[] { _classCount }, DataType.Float);
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(model, output);
            var predictionError = CNTKLib.ClassificationError(model, output);

            Func<float, TrainingParameterScheduleDouble> createTrainingParam =
                (value) => new TrainingParameterScheduleDouble(value, 0);
            IList<Learner> learners = new List<Learner>()
            {
                Learner.MomentumSGDLearner(
                    model.Parameters(),
                    createTrainingParam(learningRate),
                    createTrainingParam(momentum),
                    true,
                    new AdditionalLearningOptions()
                    {
                        l2RegularizationWeight = l2regularization
                    })
            };
            var trainer = Trainer.CreateTrainer(model, trainingLoss, predictionError, learners);

            var dataSource = InitDataSource(dataset, _trainDataFile);

            var currentMinibatch = 0;
            uint minibatchSize = 15;
            while (true)
            {
                var minibatchData = dataSource.MinibatchSource.GetNextMinibatch(minibatchSize, _device);
                var arguments = new Dictionary<Variable, MinibatchData>
                    {
                        { input, minibatchData[dataSource.FeatureStreamInfo] },
                        { output, minibatchData[dataSource.LabelStreamInfo] }
                    };
                trainer.TrainMinibatch(arguments, _device);
                CntkHelper.PrintTrainingProgress(trainer, currentMinibatch, 1);
                currentMinibatch++;
                if (currentMinibatch >= maxMinibatches)
                {
                    break;
                }
            }
                        
            Test(model, Path.Combine(_baseDataFolder, "Test"), _inputShape, _classCount);

            Console.ReadLine();
        }

        private static void Test(Function model, string testDataFolder,
            int[] imageDims, int numClasses)
        {
            var testFolder = Path.Combine(_baseDataFolder, _testFolderPrefix);
            var dataset = _datasetCreator.GetDataset(testFolder);
            
            var input = model.Arguments[0];
            var output = model.Output;

            int correct = 0, total = 0;

            var dataSource = InitDataSource(dataset, _testDataFile);
            var currentMinibatch = 0;
            uint minibatchSize = 1;
            while (true)
            {
                var minibatchData = dataSource.MinibatchSource.GetNextMinibatch(minibatchSize, _device);
                var inputDataMap = new Dictionary<Variable, Value>() { { input, minibatchData[dataSource.FeatureStreamInfo].data } };
                var outputDataMap = new Dictionary<Variable, Value>() { { output, null } };

                model.Evaluate(inputDataMap, outputDataMap, _device);
                var outputVal = outputDataMap[output];
                var actual = outputVal.GetDenseData<float>(output);
                var labelBatch = minibatchData[dataSource.LabelStreamInfo].data;
                var expected = labelBatch.GetDenseData<float>(model.Output);

                Func<IEnumerable<IList<float>>, IEnumerable<int>> maxSelector =
                    (collection) => collection.Select(x => x.IndexOf(x.Max()));

                var actualLabels = maxSelector(actual);
                var expectedLabels = maxSelector(expected);
                
                correct += actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 1 : 0).Sum();
                total += actualLabels.Count();

                currentMinibatch++;
                if (minibatchData.Values.Any(x => x.sweepEnd))
                {
                    break;
                }
            }
            var error = 1.0 * correct / dataset.Items.Count();
            Console.WriteLine($"Testing result: correctly answered {correct} of {total}, accuracy = {error * 100}%");
            
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
