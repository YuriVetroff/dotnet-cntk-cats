using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CatsClassification
{
    public class CntkModelWrapper
    {
        public CntkModelWrapper(string modelFile, DeviceDescriptor device)
        {
            Model = Function.Load(modelFile, device);

            Func<IEnumerable<int>, int> computeTotalLength =
                (dimensions) => dimensions.Aggregate(1, (a, b) => a * b);

            Input = Model.Arguments[0];
            InputShape = Input.Shape.Dimensions.ToArray();
            InputLength = computeTotalLength(InputShape);

            EvaluationOutput = Model.Output;
            OutputShape = EvaluationOutput.Shape.Dimensions.ToArray();
            OutputLength = computeTotalLength(OutputShape);

            TrainingOutput = Variable.InputVariable(OutputShape, DataType.Float);
        }

        public Function Model { get; }

        public Variable Input { get; }
        public Variable TrainingOutput { get; }
        public Variable EvaluationOutput { get; }

        public int[] InputShape { get; }
        public int InputLength { get; }

        public int[] OutputShape { get; }
        public int OutputLength { get; }
    }
}
