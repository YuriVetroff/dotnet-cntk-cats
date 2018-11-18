using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CatsClassification
{
    public class CatsClassificationRunner
    {
        private const string FEATURE_STREAM_NAME = "features";
        private const string LABEL_STREAM_NAME = "labels";

        private const int IMAGE_WIDTH = 224;
        private const int IMAGE_HEIGHT = 224;
        private const int IMAGE_DEPTH = 3;
        private const int IMAGE_TOTAL_LENGTH = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH;

        private const int CLASS_COUNT = 3;

        private readonly DeviceDescriptor _device;

        private readonly Function _model;
        private readonly Variable _input;

        public CatsClassificationRunner(string modelFile, DeviceDescriptor device)
        {
            _device = device;
            _model = Function.Load(modelFile, device);
            _input = _model.Arguments[0];
        }

        public void Train(string datasetFile)
        {
            var dataSource = CreateDataSource(datasetFile);
            var trainingOutput =
                Variable.InputVariable(new int[] { CLASS_COUNT }, DataType.Float);
            var trainer = CreateTrainer(trainingOutput);

            const int minibatchSize = 15;
            const int maxMinibatches = 5;
            var currentMinibatch = 0;
            while (true)
            {
                var minibatchData = dataSource.MinibatchSource.GetNextMinibatch(minibatchSize, _device);
                var arguments = new Dictionary<Variable, MinibatchData>
                    {
                        { _input, minibatchData[dataSource.FeatureStreamInfo] },
                        { trainingOutput, minibatchData[dataSource.LabelStreamInfo] }
                    };
                trainer.TrainMinibatch(arguments, _device);

                CntkHelper.PrintTrainingProgress(trainer, currentMinibatch++, 1);
                if (currentMinibatch >= maxMinibatches)
                {
                    break;
                }
            }
        }
        private Trainer CreateTrainer(Variable trainingOutput)
        {
            const float learningRate = 0.2F;
            const float momentum = 0.9F;
            const float l2regularization = 0.1F;
            
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(_model, trainingOutput);
            var predictionError = CNTKLib.ClassificationError(_model, trainingOutput);

            Func<float, TrainingParameterScheduleDouble> createTrainingParam =
                (value) => new TrainingParameterScheduleDouble(value, 0);
            var learners = new List<Learner>()
            {
                Learner.MomentumSGDLearner(
                    _model.Parameters(),
                    createTrainingParam(learningRate),
                    createTrainingParam(momentum),
                    true,
                    new AdditionalLearningOptions()
                    {
                        l2RegularizationWeight = l2regularization
                    })
            };

            return Trainer.CreateTrainer(
                _model,
                trainingLoss,
                predictionError,
                learners);
        }

        public void Test(string datasetFile)
        {
            var dataSource = CreateDataSource(datasetFile);
            var testingOutput = _model.Output;

            var correct = 0;
            var total = 0;

            const int minibatchSize = 1;
            var currentMinibatch = 0;
            while (true)
            {
                var minibatchData = dataSource.MinibatchSource.GetNextMinibatch(minibatchSize, _device);
                var inputDataMap = new Dictionary<Variable, Value>() { { _input, minibatchData[dataSource.FeatureStreamInfo].data } };
                var outputDataMap = new Dictionary<Variable, Value>() { { testingOutput, null } };

                _model.Evaluate(inputDataMap, outputDataMap, _device);
                var outputVal = outputDataMap[testingOutput];
                var actual = outputVal.GetDenseData<float>(testingOutput);
                var labelBatch = minibatchData[dataSource.LabelStreamInfo].data;
                var expected = labelBatch.GetDenseData<float>(_model.Output);

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
            var error = 1.0 * correct / total;
            Console.WriteLine($"Testing result: correctly answered {correct} of {total}, accuracy = {error * 100}%");
        }

        private CntkDataSource CreateDataSource(string datasetFile)
        {
            var streamConfigs = new StreamConfiguration[]
            {
                new StreamConfiguration(FEATURE_STREAM_NAME, IMAGE_TOTAL_LENGTH),
                new StreamConfiguration(LABEL_STREAM_NAME, CLASS_COUNT)
            };

            return new CntkDataSource(MinibatchSource.TextFormatMinibatchSource(
                datasetFile,
                streamConfigs,
                MinibatchSource.InfinitelyRepeat), FEATURE_STREAM_NAME, LABEL_STREAM_NAME);
        }
    }
}
