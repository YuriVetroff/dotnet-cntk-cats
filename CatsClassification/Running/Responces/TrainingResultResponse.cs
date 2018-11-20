namespace CatsClassification
{
    public class TrainingResultResponse : IRunnerResponse
    {
        public byte[] NewModelData { get; set; }
        public TrainingProgressResponse Progress { get; set; }
    }
}
