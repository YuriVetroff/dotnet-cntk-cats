using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CatsClassification
{
    public enum Activation
    {
        None,
        ReLU,
        Sigmoid,
        Tanh
    }
    public static class TestCommon
    {
        public static string TestDataDirPrefix;
    }
    public class CntkHelper
    {
        public static Function Dense(Variable input, int outputDim, DeviceDescriptor device,
            Activation activation = Activation.None, string outputName = "")
        {
            if (input.Shape.Rank != 1)
            {
                int newDim = input.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                input = CNTKLib.Reshape(input, new int[] { newDim });
            }

            Function fullyConnected = FullyConnectedLinearLayer(input, outputDim, device, outputName);
            switch (activation)
            {
                default:
                case Activation.None:
                    return fullyConnected;
                case Activation.ReLU:
                    return CNTKLib.ReLU(fullyConnected);
                case Activation.Sigmoid:
                    return CNTKLib.Sigmoid(fullyConnected);
                case Activation.Tanh:
                    return CNTKLib.Tanh(fullyConnected);
            }
        }

        public static Function FullyConnectedLinearLayer(Variable input, int outputDim, DeviceDescriptor device,
            string outputName = "")
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
            int inputDim = input.Shape[0];

            int[] s = { outputDim, inputDim };
            var timesParam = new Parameter((NDShape)s, DataType.Float,
                CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1),
                device, "timesParam");
            var timesFunction = CNTKLib.Times(timesParam, input, "times");

            int[] s2 = { outputDim };
            var plusParam = new Parameter(s2, 0.0f, device, "plusParam");
            return CNTKLib.Plus(plusParam, timesFunction, outputName);
        }

        public static void PrintTrainingProgress(Trainer trainer, int minibatchIdx, int outputFrequencyInMinibatches)
        {
            if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
            {
                var trainLossValue = trainer.PreviousMinibatchLossAverage();
                var evaluationValue = trainer.PreviousMinibatchEvaluationAverage();
                Console.WriteLine($"Minibatch: {minibatchIdx} CrossEntropyLoss = {trainLossValue}, EvaluationCriterion = {evaluationValue}");
            }
        }

        public static Function GetModel(string baseModelFile, string featureNodeName, string outputNodeName,
            string hiddenNodeName, int[] imageDims, int numClasses, DeviceDescriptor device)
        {
            Function baseModel = Function.Load(baseModelFile, device);

            var input = Variable.InputVariable(imageDims, DataType.Float);
            Function normalizedFeatureNode = CNTKLib.Minus(input, Constant.Scalar(DataType.Float, 114.0F));

            Variable oldFeatureNode = baseModel.Arguments.Single(a => a.Name == featureNodeName);
            Function lastNode = baseModel.FindByName(hiddenNodeName);

            // Clone the desired layers with fixed weights
            Function clonedLayer = CNTKLib.AsComposite(lastNode).Clone(
                ParameterCloningMethod.Freeze,
                new Dictionary<Variable, Variable>() { { oldFeatureNode, normalizedFeatureNode } });

            // Add new dense layer for class prediction
            Function clonedModel = CntkHelper.Dense(clonedLayer, numClasses, device, Activation.None, outputNodeName);

            return clonedModel;
        }
    }
}
