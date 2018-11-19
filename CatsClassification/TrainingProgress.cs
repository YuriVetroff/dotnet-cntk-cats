namespace CatsClassification
{
    public class TrainingProgress : IRunnerResponse
    {
        public int MinibatchesSeen { get; set; }
        public double? Loss { get; set; }
        public double? EvaluationCriterion { get; set; }
    }
}
