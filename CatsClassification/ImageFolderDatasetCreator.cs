using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CatsClassification
{
    public class ImageFolderDatasetCreator
    {
        private readonly IReadOnlyDictionary<string, int> _classToOutputMap;
        private readonly int _classCount;
        private readonly int _imageWidth;
        private readonly int _imageHeight;

        public ImageFolderDatasetCreator(
            IReadOnlyDictionary<string, int> classToOutputMap, int classCount, int imageWidth, int imageHeight)
        {
            _classToOutputMap = classToOutputMap;
            _classCount = classCount;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;
        }

        public Dataset GetDataset(string rootFolder) =>
            new Dataset
            {
                Items = _classToOutputMap.SelectMany(
                    entry => Directory
                        .GetFiles(Path.Combine(rootFolder, entry.Key), "*.jpg")
                        .Select(file => new DataItem
                        {
                            Input = ImageHelper.Load(_imageWidth, _imageHeight, file),
                            Output = Enumerable.Range(0, _classCount)
                                .Select(x => (float)(x == entry.Value ? 1 : 0))
                                .ToArray()
                        }))
                        .ToList()
            };
    }
}
