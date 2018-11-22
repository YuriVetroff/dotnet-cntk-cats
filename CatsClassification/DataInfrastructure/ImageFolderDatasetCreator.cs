using CatsClassification.Common;
using CatsClassification.Configuration;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CatsClassification.DataInfrastructure
{
    public class ImageFolderDatasetCreator
    {
        private readonly ClassificationConfig _config;
        private readonly int _imageWidth;
        private readonly int _imageHeight;

        public ImageFolderDatasetCreator(
            ClassificationConfig config, int imageWidth, int imageHeight)
        {
            _config = config;
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;
        }

        public Dataset GetDataset(string rootFolder) =>
            new Dataset
            {
                Items = _config.ClassNames.SelectMany(
                    className => Directory
                        .GetFiles(Path.Combine(rootFolder, className), "*.jpg")
                        .Select(file => new DataItem
                        {
                            Input = ImageHelper.Load(_imageWidth, _imageHeight, file),
                            Output = Enumerable.Range(0, _config.ClassCount)
                                .Select(x => (float)(x == _config.GetIndexByClassName(className) ? 1 : 0))
                                .ToArray()
                        }))
                        .ToList()
            };
    }
}
