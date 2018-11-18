using CNTK;

namespace CatsClassification
{
    public class CntkDataSource
    {
        public CntkDataSource(MinibatchSource source, string featureStreamName, string labelStreamName)
        {
            MinibatchSource = source;
            FeatureStreamInfo = MinibatchSource.StreamInfo(featureStreamName);
            LabelStreamInfo = MinibatchSource.StreamInfo(labelStreamName);
        }

        public StreamInformation FeatureStreamInfo { get; }
        public StreamInformation LabelStreamInfo { get; }

        public MinibatchSource MinibatchSource { get; }
    }
}
