using System.Collections.Generic;
using System.IO;
using System.Text;

namespace CatsClassification.DataInfrastructure
{
    public class DataFileCreator
    {
        private const string LABELS_KEY = "|labels";
        private const string FEATURES_KEY = "|features";

        public void CreateDataFile(Dataset dataset, string filename)
        {
            var lines = new List<string>();
            foreach (var item in dataset.Items)
            {
                var line = $"{LABELS_KEY} {CreateDataString(item.Output)} {FEATURES_KEY} {CreateDataString(item.Input)}";
                lines.Add(line.Trim());
            }

            File.WriteAllLines(filename, lines);
        }

        private string CreateDataString(float[] data)
        {
            var result = new StringBuilder();

            foreach (var x in data)
            {
                result.Append($"{x}");
                result.Append(" ");
            }

            return result.ToString();
        }
    }
}
