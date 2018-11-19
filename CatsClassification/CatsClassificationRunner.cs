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

        private readonly DeviceDescriptor _device;
        private readonly CntkModelWrapper _modelWrapper;

        public CatsClassificationRunner(string modelFile, DeviceDescriptor device)
        {
            _device = device;
            _modelWrapper = new CntkModelWrapper(modelFile, device);
        }

        public void Train(string datasetFile)
        {
            var dataSource = CreateDataSource(datasetFile);
            var trainer = CreateTrainer();

            const int minibatchSize = 15;
            const int maxMinibatches = 5;
            var currentMinibatch = 0;
            while (true)
            {
                var minibatchData = dataSource.MinibatchSource.GetNextMinibatch(minibatchSize, _device);
                var arguments = new Dictionary<Variable, MinibatchData>
                    {
                        { _modelWrapper.Input, minibatchData[dataSource.FeatureStreamInfo] },
                        { _modelWrapper.TrainingOutput, minibatchData[dataSource.LabelStreamInfo] }
                    };
                trainer.TrainMinibatch(arguments, _device);

                CntkHelper.PrintTrainingProgress(trainer, currentMinibatch++, 1);
                if (currentMinibatch >= maxMinibatches)
                {
                    break;
                }
            }
        }
        private Trainer CreateTrainer()
        {
            const float learningRate = 0.2F;
            const float momentum = 0.9F;
            const float l2regularization = 0.1F;
            
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(_modelWrapper.Model, _modelWrapper.TrainingOutput);
            var predictionError = CNTKLib.ClassificationError(_modelWrapper.Model, _modelWrapper.TrainingOutput);

            Func<float, TrainingParameterScheduleDouble> createTrainingParam =
                (value) => new TrainingParameterScheduleDouble(value, 0);
            var learners = new List<Learner>()
            {
                Learner.MomentumSGDLearner(
                    _modelWrapper.Model.Parameters(),
                    createTrainingParam(learningRate),
                    createTrainingParam(momentum),
                    true,
                    new AdditionalLearningOptions()
                    {
                        l2RegularizationWeight = l2regularization
                    })
            };

            return Trainer.CreateTrainer(
                _modelWrapper.Model,
                trainingLoss,
                predictionError,
                learners);
        }

        public void Test(string datasetFile)
        {
            var dataSource = CreateDataSource(datasetFile);

            var correct = 0;
            var total = 0;

            const int minibatchSize = 1;
            var currentMinibatch = 0;
            while (true)
            {
                var minibatchData = dataSource.MinibatchSource.GetNextMinibatch(minibatchSize, _device);
                var inputDataMap = new Dictionary<Variable, Value>() { { _modelWrapper.Input, minibatchData[dataSource.FeatureStreamInfo].data } };
                var outputDataMap = new Dictionary<Variable, Value>() { { _modelWrapper.EvaluationOutput, null } };

                _modelWrapper.Model.Evaluate(inputDataMap, outputDataMap, _device);
                var outputVal = outputDataMap[_modelWrapper.EvaluationOutput];
                var actual = outputVal.GetDenseData<float>(_modelWrapper.EvaluationOutput);
                var labelBatch = minibatchData[dataSource.LabelStreamInfo].data;
                var expected = labelBatch.GetDenseData<float>(_modelWrapper.Model.Output);

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

        private CntkDataSource CreateDataSource(string datasetFile) =>
            new CntkDataSource(
                MinibatchSource.TextFormatMinibatchSource(
                    datasetFile,
                    new StreamConfiguration[]
                    {
                        new StreamConfiguration(FEATURE_STREAM_NAME, _modelWrapper.InputLength),
                        new StreamConfiguration(LABEL_STREAM_NAME, _modelWrapper.OutputLength)
                    },
                    MinibatchSource.InfinitelyRepeat),
                FEATURE_STREAM_NAME,
                LABEL_STREAM_NAME);
    }
}
