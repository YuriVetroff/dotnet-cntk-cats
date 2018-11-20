using CatsClassification.Running.Domain;

namespace CatsClassification.Running
{
    public static class RunnerFactory
    {
        public static IRunner GetRunnerForCats() =>
            new CatsClassificationRunner();
    }
}
