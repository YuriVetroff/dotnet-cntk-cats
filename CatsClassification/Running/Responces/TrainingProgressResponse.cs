namespace CatsClassification.Running.Responses
{
    public class TrainingProgressResponse : IRunnerResponse
    {
        public int MinibatchesSeen { get; set; }
        public double? Loss { get; set; }
        public double? EvaluationCriterion { get; set; }
    }
}
