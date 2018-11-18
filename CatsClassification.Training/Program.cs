using System;
using System.Collections.Generic;

namespace CatsClassification.Training
{
    internal class Program
    {
        private const string TRAIN_DATASET_FILE = "D:/train-dataset.txt";
        private const string TEST_DATASET_FILE = "D:/test-dataset.txt";

        private static void Main(string[] args)
        {
            Console.WriteLine("Cats classification");
            TransferLearning.Train();
        }
    }
}
