using System;

namespace CatsClassification
{
    public abstract class AbstractRunner : IRunner
    {
        public abstract void Train(string datasetFile);
        public abstract void Test(string datasetFile);

        public event EventHandler<TrainingProgress> TrainingIterationPerformed;
        public event EventHandler<TrainingResult> TrainingFinished;
        public event EventHandler<TestingResult> TestingFinished;

        protected void OnTrainingIterationPerformed(TrainingProgress trainingProgress)
            => TrainingIterationPerformed?.Invoke(this, trainingProgress);
        protected void OnTrainingFinished(TrainingResult trainingResult)
            => TrainingFinished?.Invoke(this, trainingResult);
        protected void OnTestingFinished(TestingResult testingResult)
            => TestingFinished?.Invoke(this, testingResult);
    }
}
