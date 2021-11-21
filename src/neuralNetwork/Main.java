package neuralNetwork;

import java.util.List;

public class Main {
    public static void main(String[] args) {
//        TrainingSet trainingSet = Input3ANDgAte();
        TrainingSet trainingSet = XOR();
        neuralNetwork.NeuralNetwork nn = new neuralNetwork.NeuralNetwork(trainingSet.tasks[0].length, 10, 1);
        List<Double> output;


        // train the nn
        nn.fit(trainingSet.tasks, trainingSet.targets, 500000);

        try {
            nn= NeuralNetwork.getNNFromJson("NN.json");
        } catch (Exception e) {
            e.printStackTrace();
        }
        // test set
        double[][] input = {
                {0, 0, 0}, {0, 1, 1}, {1, 0, 0}, {1, 1, 1}
        };
        // print the outputs
        for (double[] d : input) {
            output = nn.predict(d);
            System.out.println(output.toString());
        }
        // print weights
        System.out.println(nn.weights_ho.toArray().toString());
        System.out.println(nn.weights_ih.toArray().toString());
        try {
            nn.parseToJson("./NN.json");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static TrainingSet XOR() {
        // learning set:
        // XOR Gate:
//     | ===================|
//     | A   B   |  Output  |
//     |--------------------|
//     | 0   0   |    0     |
//     | 0   1   |    1     |
//     | 1   0   |    1     |
//     | 1   1   |    0     |
//     | ===================|

        double[][] X = {
                {0, 0},
                {1, 0},
                {0, 1},
                {1, 1}
        };
        // solutions for the learning set
        double[][] Y = {
                {0}, {1}, {1}, {0}
        };
        return new TrainingSet(X, Y);
    }

    private static TrainingSet Input3ANDgAte() {
        // learning set:
        // 3-input-AND Gate:
//     | =======================|
//     | A   B   C   |  Output  |
//     |------------------------|
//     | 0   0   0   |    0     |
//     | 0   0   1   |    1     |
//     | 0   1   0   |    1     |
//     | 0   1   1   |    1     |
//     | 1   0   0   |    1     |
//     | 1   0   1   |    1     |
//     | 1   1   0   |    1     |
//     | 1   1   1   |    1     |
//     | =======================|

        double[][] X = {
                {0, 0, 0},
                {0, 0, 1},
                {0, 1, 0},
                {0, 1, 1},
                {1, 0, 0},
                {1, 0, 1},
                {1, 1, 0},
                {1, 1, 1},
        };
        // solutions for the learning set
        double[][] Y = {
                {0}, {1}, {1}, {1}, {1}, {1}, {1}, {1}
        };
        return new TrainingSet(X, Y);
    }
}