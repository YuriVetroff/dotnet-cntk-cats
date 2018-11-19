namespace CatsClassification
{
    public class TrainingResult : IRunnerResponse
    {
        public byte[] NewModelData { get; set; }
        public TrainingProgress Progress { get; set; }
    }
}
