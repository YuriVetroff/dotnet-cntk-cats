using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CatsClassification.Configuration
{
    public class ClassificationConfig
    {
        private readonly string[] _classNames;

        public ClassificationConfig(IEnumerable<string> classNames)
        {
            _classNames = classNames.ToArray();
        }

        public IEnumerable<string> ClassNames => _classNames;
        public int ClassCount => _classNames.Length;

        public int GetIndexByClassName(string className) =>
            Array.IndexOf(_classNames, className);

        public string GetClassNameByIndex(int index) =>
            _classNames[index];

        public void Save(
            string filename) =>
            File.WriteAllLines(
                filename, _classNames);

        public static ClassificationConfig Load(
            string filename) =>
            new ClassificationConfig(
                File.ReadAllLines(filename));
    }
}
