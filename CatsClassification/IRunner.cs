using System;

namespace CatsClassification
{
    public interface IRunner
    {
        void Mount(string modelFile);
        void Train(string datasetFile);
        void Test(string datasetFile);

        event EventHandler<TrainingProgress> TrainingIterationPerformed;
        event EventHandler<TrainingResult> TrainingFinished;
        event EventHandler<TestingResult> TestingFinished;
    }
}
