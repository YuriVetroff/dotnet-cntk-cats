using System;

namespace CatsClassification
{
    public interface IRunner
    {
        void Mount(string modelFile);
        void Train(string datasetFile);
        void Test(string datasetFile);

        event EventHandler<TrainingProgressResponse> TrainingIterationPerformed;
        event EventHandler<TrainingResultResponse> TrainingFinished;
        event EventHandler<TestingResultResponse> TestingFinished;
    }
}
