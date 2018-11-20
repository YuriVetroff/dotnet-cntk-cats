using System;

namespace CatsClassification
{
    public interface IRunner
    {
        void Train(string datasetFile);
        void Test(string datasetFile);

        event EventHandler<TrainingProgress> TrainingIterationPerformed;
        event EventHandler<TrainingResult> TrainingFinished;
        event EventHandler<TestingResult> TestingFinished;
    }
}
