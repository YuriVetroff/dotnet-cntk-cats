using CNTK;
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

        public static Function BuildTransferLearningModel(Function baseModel, string featureNodeName, string outputNodeName,
            string hiddenNodeName, int[] imageDims, int numClasses, DeviceDescriptor device)
        {
            var input = Variable.InputVariable(imageDims, DataType.Float);
            var normalizedFeatureNode = CNTKLib.Minus(input, Constant.Scalar(DataType.Float, 114.0F));

            var oldFeatureNode = baseModel.Arguments.Single(a => a.Name == featureNodeName);
            var lastNode = baseModel.FindByName(hiddenNodeName);
            
            var clonedLayer = CNTKLib.AsComposite(lastNode).Clone(
                ParameterCloningMethod.Freeze,
                new Dictionary<Variable, Variable>()
                {
                    { oldFeatureNode, normalizedFeatureNode }
                });
            
            var clonedModel = Dense(clonedLayer, numClasses, device, Activation.None, outputNodeName);
            return clonedModel;
        }
    }
}
