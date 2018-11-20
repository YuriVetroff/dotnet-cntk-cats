using System.Collections.Generic;

namespace CatsClassification.DataInfrastructure
{
    public class Dataset
    {
        public IReadOnlyList<DataItem> Items { get; set; }
    }
}
