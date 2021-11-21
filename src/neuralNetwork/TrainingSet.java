package neuralNetwork;

public class TrainingSet {
    double[][] tasks;
    double[][] targets;

    public TrainingSet(double[][] X, double[][] Y) {
        this.tasks = X;
        this.targets = Y;
    }
}
