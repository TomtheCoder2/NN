package neuralNetwork;

import com.google.gson.*;

import java.io.*;
import java.util.List;

/**
 * Class to generate and save Neural Networks with one hidden layer
 */
public class NeuralNetwork {
    Matrix weights_ih, // weights for input and hidden layer
            weights_ho, // weights for hidden and output layer
            bias_h, // bias hidden layer
            bias_o; // bias output layer
    double l_rate = 0.01; // learning rate

    /**
     * Create a new Neural Network
     *
     * @param i amount of input nodes
     * @param h amount of hidden nodes
     * @param o amount of outputs
     */
    public NeuralNetwork(int i, int h, int o) {
        weights_ih = new Matrix(h, i);
        weights_ho = new Matrix(o, h);

        bias_h = new Matrix(h, 1);
        bias_o = new Matrix(o, 1);
    }

    /**
     * Predict a result based on current weights and biases
     */
    public List<Double> predict(double[] X) {
        Matrix input = Matrix.fromArray(X);
        Matrix hidden = Matrix.multiply(weights_ih, input);
        hidden.add(bias_h);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weights_ho, hidden);
        output.add(bias_o);
        output.sigmoid();

        return output.toArray();
    }

    /**
     * One training iteration
     */
    public void train(double[] X, double[] Y) {
        // predict the output

        // convert the input to an array
        Matrix input = Matrix.fromArray(X);

        // first layer:
        Matrix hidden = Matrix.multiply(weights_ih, input);
        hidden.add(bias_h); // add the bias
        hidden.sigmoid(); // run sigmoid to get an output 0 / 1

        // second layer:
        Matrix output = Matrix.multiply(weights_ho, hidden);
        output.add(bias_o); // add the bias
        output.sigmoid(); // run sigmoid to get an output 0 / 1


        // error detection and correction
        Matrix target = Matrix.fromArray(Y);

        // calculate the error between the output and the correct output (target)
        Matrix error = Matrix.subtract(target, output);
        // use the derivative sigmoid function to correct the error
        Matrix gradient = output.dsigmoid();
        gradient.multiply(error);
        gradient.multiply(l_rate);

        // backpropagation
        Matrix hidden_T = Matrix.transpose(hidden);
        Matrix who_delta = Matrix.multiply(gradient, hidden_T);

        // apply correction
        weights_ho.add(who_delta);
        bias_o.add(gradient);

        // calculate the errors in the hidden layers
        Matrix who_T = Matrix.transpose(weights_ho);
        Matrix hidden_errors = Matrix.multiply(who_T, error);

        // sigmoid + dev
        Matrix h_gradient = hidden.dsigmoid();
        h_gradient.multiply(hidden_errors);
        h_gradient.multiply(l_rate);

        Matrix i_T = Matrix.transpose(input);
        Matrix wih_delta = Matrix.multiply(h_gradient, i_T);

        weights_ih.add(wih_delta);
        bias_h.add(h_gradient);
    }

    /**
     * Train the Neural Network
     *
     * @param X list of samples (training set)
     * @param Y solutions of the training set X
     */
    public void fit(double[][] X, double[][] Y, int epochs) {
        for (int i = 0; i < epochs; i++) {
            int sampleN = (int) (Math.random() * X.length);
            this.train(X[sampleN], Y[sampleN]);
        }
    }

    public void parseToJson(String path) throws IOException {
        Gson gson = new Gson();
        FileWriter writer = new FileWriter(path);
        gson.toJson(this, writer);
        writer.flush(); // flush data to file   <---
        writer.close(); // close write          <---
        System.out.println(gson.toJson(this));
    }

    public static NeuralNetwork getNNFromJson(String path) throws FileNotFoundException {
        Gson gson = new Gson();
        return gson.fromJson(new FileReader(path), NeuralNetwork.class);
    }
}
