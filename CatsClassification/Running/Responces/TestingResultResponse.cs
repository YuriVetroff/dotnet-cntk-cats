﻿namespace CatsClassification.Running.Responses
{
    public class TestingResultResponse : IRunnerResponse
    {
        public int Correct { get; set; }
        public int Total { get; set; }
        public double Accuracy => 1d * Correct / Total;
    }
}
