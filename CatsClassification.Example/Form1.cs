using CatsClassification.Common;
using CatsClassification.Running;
using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

namespace CatsClassification.Example
{
    public partial class Form1 : Form
    {
        private const string TRAINED_MODEL_FILE = "cats-classification/cats-classifier-trained.model";
        private readonly CntkModelWrapper _modelWrapper;
        private readonly DeviceDescriptor _device = DeviceDescriptor.CPUDevice;
        private string _imageFile = "";
        private readonly IReadOnlyDictionary<int, string> _outputToClassMap = new Dictionary<int, string>()
        {
            { 0, "Tiger" },
            { 1, "Leopard" },
            { 2, "Puma" }
        };

        public Form1()
        {
            InitializeComponent();
            _modelWrapper = new CntkModelWrapper(TRAINED_MODEL_FILE, _device);
        }

        private void buttonOpen_Click(object sender, EventArgs e)
        {
            using (var dialog = new OpenFileDialog())
            {
                dialog.Title = "Open Image";
                dialog.Filter = "jpg files (*.jpg)|*.jpg";

                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    _imageFile = dialog.FileName;
                    pictureBoxMain.Image = new Bitmap(dialog.FileName);
                }
            }
        }

        private void buttonClassify_Click(object sender, EventArgs e)
        {
            var inputValue = new Value(new NDArrayView(new int[] { 224, 224, 3 }, ImageHelper.Load(224, 224, _imageFile), _device));
            var inputDataMap = new Dictionary<Variable, Value>() { { _modelWrapper.Input, inputValue } };
            var outputDataMap = new Dictionary<Variable, Value>() { { _modelWrapper.EvaluationOutput, null } };

            _modelWrapper.Model.Evaluate(inputDataMap, outputDataMap, _device);
            var outputData = outputDataMap[_modelWrapper.EvaluationOutput].GetDenseData<float>(_modelWrapper.EvaluationOutput).First();

            var output = outputData.Select(x => (double)x).ToArray();
            var classIndex = Array.IndexOf(output, output.Max());
            var className = _outputToClassMap[classIndex];

            labelClass.Text = className;
        }
    }
}
