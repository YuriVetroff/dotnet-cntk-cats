using System;

namespace CatsClassification
{
    public abstract class AbstractRunner : IRunner
    {
        public abstract void Mount(string modelFile);
        public abstract void Train(string datasetFile);
        public abstract void Test(string datasetFile);

        public event EventHandler<TrainingProgressResponse> TrainingIterationPerformed;
        public event EventHandler<TrainingResultResponse> TrainingFinished;
        public event EventHandler<TestingResultResponse> TestingFinished;

        protected void OnTrainingIterationPerformed(TrainingProgressResponse trainingProgressResponse)
            => TrainingIterationPerformed?.Invoke(this, trainingProgressResponse);
        protected void OnTrainingFinished(TrainingResultResponse trainingResultResponse)
            => TrainingFinished?.Invoke(this, trainingResultResponse);
        protected void OnTestingFinished(TestingResultResponse testingResultResponse)
            => TestingFinished?.Invoke(this, testingResultResponse);
    }
}
