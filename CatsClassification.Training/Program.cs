using CNTK;
using System;

namespace CatsClassification.Training
{
    class Program
    {
        static void Main(string[] args)
        {
            TestCommon.TestDataDirPrefix = "D:/";
            var device = DeviceDescriptor.CPUDevice;

            Console.WriteLine("======== running TransferLearning.TrainAndEvaluateWithAnimalData using CPU ========");
            TransferLearning.Train(device);
        }
    }
}
