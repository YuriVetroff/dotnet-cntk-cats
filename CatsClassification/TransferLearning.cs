using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace CatsClassification
{
    public class TransferLearning
    {
        public static string CurrentFolder = "D:/";

        public static string ExampleImageFolder
        {
            get => TestCommon.TestDataDirPrefix;
            set => TestCommon.TestDataDirPrefix = value;
        }

        public static string BaseResnetModelFile = TestCommon.TestDataDirPrefix + "/ResNet18_ImageNet_CNTK.model";

        private static string featureNodeName = "features";
        private static string lastHiddenNodeName = "z.x";
        private static int[] imageDims = new int[] { 224, 224, 3 };

        private static string _baseDataFolder = Path.Combine(ExampleImageFolder, "Datasets/Animals-cats");
        private static string _trainFolderPrefix = "Train";
        private static string _testFolderPrefix = "Test";
        private static ImageFolderDatasetCreator _datasetCreator =
            new ImageFolderDatasetCreator(new Dictionary<string, int>
            {
                { "Tiger", 0 },
                { "Leopard", 1 },
                { "Puma", 2 },
            }, 3, imageDims[0], imageDims[1]);

        public static void Train(DeviceDescriptor device, bool forceRetrain = true)
        {
            int numClasses = 3;
            string animalsModelFile = Path.Combine(CurrentFolder, "AnimalsTransferLearning.model");

            var trainFolder = Path.Combine(_baseDataFolder, _trainFolderPrefix);
            var dataset = _datasetCreator.GetDataset(trainFolder);

            // prepare the transfer model
            string predictionNodeName = "prediction";
            Variable imageInput, labelInput;
            Function model = CntkHelper.GetModel(
                Path.Combine(ExampleImageFolder, BaseResnetModelFile),
                featureNodeName,
                predictionNodeName,
                lastHiddenNodeName,
                imageDims,
                numClasses,
                device,
                out imageInput, out labelInput);

            var input = Variable.InputVariable(imageDims, DataType.Float);
            var labels = Variable.InputVariable(new int[] { numClasses }, DataType.Float);


            // prepare for training
            int maxMinibatches = 5;
            float learningRate = 0.2F;
            float momentum = 0.9F;
            float l2regularization = 0.1F;

            AdditionalLearningOptions additionalLearningOptions =
                new AdditionalLearningOptions() { l2RegularizationWeight = l2regularization };
            IList<Learner> parameterLearners = new List<Learner>() {
                    Learner.MomentumSGDLearner(model.Parameters(),
                    new TrainingParameterScheduleDouble(learningRate, 0),
                    new TrainingParameterScheduleDouble(momentum, 0),
                    true,
                    additionalLearningOptions)};
            

            var output = Variable.InputVariable(new int[] { numClasses }, DataType.Float);
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(model, output);
            var predictionError = CNTKLib.ClassificationError(model, output);
            var trainer = Trainer.CreateTrainer(model, trainingLoss, predictionError, parameterLearners);

            for (int minibatchCount = 0; minibatchCount < maxMinibatches; ++minibatchCount)
            {
                Value imageBatch, labelBatch;
                int batchCount = 0, batchSize = 15;
                while (GetMinibatch(dataset, batchSize, batchCount++,
                    imageDims, numClasses, device, out imageBatch, out labelBatch))
                {
#pragma warning disable 618
                    trainer.TrainMinibatch(
                        new Dictionary<Variable, Value>()
                        {
                            { imageInput, imageBatch },
                            { output, labelBatch }
                        },
                        device);
#pragma warning restore 618
                    CntkHelper.PrintTrainingProgress(trainer, minibatchCount, 1);
                }
            }

            // save the trained model
            model.Save(animalsModelFile);

            // done with training, continue with validation
            double error = Test(
                animalsModelFile,
                Path.Combine(_baseDataFolder, "Test"),
                imageDims,
                numClasses,
                device);

            Console.ReadLine();
        }

        private static bool GetMinibatch(Dataset dataset,
            int batchSize, int batchCount, int[] imageDims, int numClasses, DeviceDescriptor device,
            out Value imageBatch, out Value labelBatch)
        {
            int actualBatchSize = Math.Min(dataset.Items.Count() - batchSize * batchCount, batchSize);
            if (actualBatchSize <= 0)
            {
                imageBatch = null;
                labelBatch = null;
                return false;
            }

            if (batchCount == 0)
            {
                var random = new Random(0);
                dataset.Items = dataset.Items.OrderBy(x => random.Next()).ToList();
            }

            int imageSize = imageDims[0] * imageDims[1] * imageDims[2];
            float[] batchImageBuf = new float[actualBatchSize * imageSize];
            float[] batchLabelBuf = new float[actualBatchSize * numClasses];
            for (int i = 0; i < actualBatchSize; i++)
            {
                int index = i + batchSize * batchCount;
                dataset.Items[index].Input.CopyTo(batchImageBuf, i * imageSize);
                dataset.Items[index].Output.CopyTo(batchLabelBuf, i * numClasses);
            }

            imageBatch = Value.CreateBatch(imageDims, batchImageBuf, device);
            labelBatch = Value.CreateBatch(new int[] { numClasses }, batchLabelBuf, device);
            return true;
        }

        private static double Test(string modelFile, string testDataFolder,
            int[] imageDims, int numClasses, DeviceDescriptor device)
        {
            var testFolder = Path.Combine(_baseDataFolder, _testFolderPrefix);
            var dataset = _datasetCreator.GetDataset(testFolder);

            Function model = Function.Load(modelFile, device);
            Value imageBatch, labelBatch;
            int batchCount = 0, batchSize = 15;
            int miscountTotal = 0, totalCount = 0;
            while (GetMinibatch(dataset, batchSize, batchCount++,
                TransferLearning.imageDims, numClasses, device, out imageBatch, out labelBatch))
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { model.Arguments[0], imageBatch } };

                Variable outputVar = model.Output;
                var outputDataMap = new Dictionary<Variable, Value>() { { outputVar, null } };
                model.Evaluate(inputDataMap, outputDataMap, device);
                var outputVal = outputDataMap[outputVar];
                var actual = outputVal.GetDenseData<float>(outputVar);
                var expected = labelBatch.GetDenseData<float>(model.Output);

                var actualLabels = actual.Select((IList<float> l) => l.IndexOf(l.Max())).ToList();
                var expectedLabels = expected.Select((IList<float> l) => l.IndexOf(l.Max())).ToList();

                int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();
                miscountTotal += misMatches;
                totalCount += actualLabels.Count();

                Console.WriteLine($"Validating Model: Total Samples = {totalCount}, Misclassify Count = {miscountTotal}");
            }

            Console.WriteLine(miscountTotal);
            return 1.0 * miscountTotal / dataset.Items.Count();
        }

        private static Dictionary<string, int> LoadMapFile(string mapFile)
        {
            Dictionary<string, int> imageFileToLabel = new Dictionary<string, int>();
            string line;

            if (File.Exists(mapFile))
            {
                StreamReader file = null;
                try
                {
                    file = new StreamReader(mapFile);
                    while ((line = file.ReadLine()) != null)
                    {
                        int spaceIndex = line.IndexOfAny(new char[] { ' ', '\t' });
                        string filePath = line.Substring(0, spaceIndex);
                        int label = int.Parse(line.Substring(spaceIndex).Trim());
                        imageFileToLabel.Add(filePath, label);
                    }
                }
                finally
                {
                    if (file != null)
                        file.Close();
                }
            }
            return imageFileToLabel;
        }
    }
}
