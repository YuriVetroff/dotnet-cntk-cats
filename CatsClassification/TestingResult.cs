namespace CatsClassification
{
    public class TestingResult : IRunnerResponse
    {
        public int Correct { get; set; }
        public int Total { get; set; }
        public double Accuracy => 1d * Correct / Total;
    }
}
