package eu.veldsoft.digits.resilient;

import java.io.FileInputStream;

import org.encog.ConsoleStatusReportable;
import org.encog.engine.network.activation.ActivationBipolarSteepenedSigmoid;
import org.encog.engine.network.activation.ActivationElliottSymmetric;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationLOG;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.neural.prune.PruneIncremental;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

public class DigitsResilient {
	private static int NUMBER_OF_EXPERIMENTS = 30;

	private static int INPUT_SIZE = 256;

	private static int HIDDEN_SIZE = 175;

	private static int OUTPUT_SIZE = 10;

	private static int PRUNE_ITERATIONS = 300;

	private static int MIN_HIDDEN = 1;

	private static int MAX_HIDDEN = INPUT_SIZE + OUTPUT_SIZE;

	private static final long MAX_TRAINING_TIME = 10000;

	private static final int MAX_EPOCHS = 1000;

	private static final double TARGET_ANN_ERROR = 0.1;

	private static final NeuralDataSet ZERO_ONE_TRAINING = new BasicNeuralDataSet();

	private static final NeuralDataSet MINUS_PLUS_ONE_TRAINING = new BasicNeuralDataSet();

	/*
	 * It is used for time measurement calibration.
	 */
	static {
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, INPUT_SIZE));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, HIDDEN_SIZE));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, OUTPUT_SIZE));
		network.getStructure().finalizeStructure();
		network.reset();

		try {
			(new ResilientPropagation(network, new BasicNeuralDataSet())).iteration();
		} catch (Exception e) {
		}

		try {
			ReadCSV csv = new ReadCSV(new FileInputStream("./data/digits1.csv"), false, CSVFormat.DECIMAL_POINT);
			while (csv.next() == true) {
				MLData input = new BasicMLData(INPUT_SIZE);
				MLData ideal = new BasicMLData(OUTPUT_SIZE);
				MLDataPair pair = new BasicMLDataPair(input, ideal);

				int index = 0;
				for (int i = 0; i < input.size(); i++) {
					input.setData(i, csv.getDouble(index++));
				}
				for (int i = 0; i < ideal.size(); i++) {
					ideal.setData(i, csv.getDouble(index++));
				}
				ZERO_ONE_TRAINING.add(pair);
			}
			csv.close();

			csv = new ReadCSV(new FileInputStream("./data/digits2.csv"), false, CSVFormat.DECIMAL_POINT);
			while (csv.next() == true) {
				MLData input = new BasicMLData(256);
				MLData ideal = new BasicMLData(10);
				MLDataPair pair = new BasicMLDataPair(input, ideal);

				int index = 0;
				for (int i = 0; i < input.size(); i++) {
					input.setData(i, csv.getDouble(index++));
				}
				for (int i = 0; i < ideal.size(); i++) {
					ideal.setData(i, csv.getDouble(index++));
				}
				MINUS_PLUS_ONE_TRAINING.add(pair);
			}
			csv.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static BasicNetwork prune(String title, ActivationFunction activation, NeuralDataSet training,
			int iterations, int hiddenMin, int hiddenMax) {
		FeedForwardPattern pattern = new FeedForwardPattern();
		pattern.setInputNeurons(training.getInputSize());
		pattern.setOutputNeurons(training.getIdealSize());
		pattern.setActivationFunction(activation);

		PruneIncremental prune = new PruneIncremental(training, pattern, iterations, 3, 10,
				new ConsoleStatusReportable());

		prune.addHiddenLayer(hiddenMin, hiddenMax);

		prune.process();

		return prune.getBestNetwork();
	}

	private static Object[] train(String title, ActivationFunction activation, NeuralDataSet training, double epsilon) {
		Object result[] = { title, Long.valueOf(0), Long.valueOf(0), Double.valueOf(0) };

		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(
				activation instanceof ActivationFadingSin ? new ActivationFadingSin(INPUT_SIZE) : activation, true,
				INPUT_SIZE));
		network.addLayer(new BasicLayer(
				activation instanceof ActivationFadingSin ? new ActivationFadingSin(HIDDEN_SIZE) : activation, true,
				HIDDEN_SIZE));
		network.addLayer(new BasicLayer(
				activation instanceof ActivationFadingSin ? new ActivationFadingSin(OUTPUT_SIZE) : activation, false,
				OUTPUT_SIZE));
		network.getStructure().finalizeStructure();
		network.reset();

		final Train train = new ResilientPropagation(network, training);

		int epoch = 1;
		long start = System.currentTimeMillis();

		do {
			train.iteration();
			epoch++;
		} while (train.getError() > epsilon && (System.currentTimeMillis() - start) < MAX_TRAINING_TIME
				&& epoch < MAX_EPOCHS);

		result[1] = (System.currentTimeMillis() - start);
		result[2] = epoch;
		result[3] = train.getError();

		for (MLDataPair pair : training) {
			final MLData output = network.compute(pair.getInput());
		}

		return result;
	}

	public static void main(final String args[]) {
		BasicNetwork net = null;
		net = prune("Fading Sine", new ActivationFadingSin(1), MINUS_PLUS_ONE_TRAINING, PRUNE_ITERATIONS, MIN_HIDDEN,
				MAX_HIDDEN);
		net = prune("Exponent Regulated Sine", new ActivationExponentRegulatedSin(1), MINUS_PLUS_ONE_TRAINING,
				PRUNE_ITERATIONS, MIN_HIDDEN, MAX_HIDDEN);
		net = prune("Sigmoid", new ActivationSigmoid(), ZERO_ONE_TRAINING, PRUNE_ITERATIONS, MIN_HIDDEN, MAX_HIDDEN);
		net = prune("Bipolar Sigmoid", new ActivationBipolarSteepenedSigmoid(), MINUS_PLUS_ONE_TRAINING,
				PRUNE_ITERATIONS, MIN_HIDDEN, MAX_HIDDEN);
		net = prune("Logarithm", new ActivationLOG(), MINUS_PLUS_ONE_TRAINING, PRUNE_ITERATIONS, MIN_HIDDEN,
				MAX_HIDDEN);
		net = prune("Hyperbolic Tangent", new ActivationTANH(), MINUS_PLUS_ONE_TRAINING, PRUNE_ITERATIONS, MIN_HIDDEN,
				MAX_HIDDEN);
		net = prune("Elliott Symmetric", new ActivationElliottSymmetric(), MINUS_PLUS_ONE_TRAINING, PRUNE_ITERATIONS,
				MIN_HIDDEN, MAX_HIDDEN);

		Object statistics[] = {};
		for (long g = 0; g < NUMBER_OF_EXPERIMENTS; g++) {
			statistics = train("Fading Sine", new ActivationFadingSin(0), MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR);
			System.out.println(statistics[0] + "\t" + statistics[1] + "\t" + statistics[2] + "\t" + statistics[3]);
			statistics = train("Exponent Regulated Sine", new ActivationExponentRegulatedSin(0),
					MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR);
			System.out.println(statistics[0] + "\t" + statistics[1] + "\t" + statistics[2] + "\t" + statistics[3]);
			statistics = train("Sigmoid", new ActivationSigmoid(), ZERO_ONE_TRAINING, TARGET_ANN_ERROR);
			System.out.println(statistics[0] + "\t" + statistics[1] + "\t" + statistics[2] + "\t" + statistics[3]);
			statistics = train("Bipolar Sigmoid", new ActivationBipolarSteepenedSigmoid(), MINUS_PLUS_ONE_TRAINING,
					TARGET_ANN_ERROR);
			System.out.println(statistics[0] + "\t" + statistics[1] + "\t" + statistics[2] + "\t" + statistics[3]);
			statistics = train("Logarithm", new ActivationLOG(), MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR);
			System.out.println(statistics[0] + "\t" + statistics[1] + "\t" + statistics[2] + "\t" + statistics[3]);
			statistics = train("Hyperbolic Tangent", new ActivationTANH(), MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR);
			System.out.println(statistics[0] + "\t" + statistics[1] + "\t" + statistics[2] + "\t" + statistics[3]);
			statistics = train("Elliott Symmetric", new ActivationElliottSymmetric(), MINUS_PLUS_ONE_TRAINING,
					TARGET_ANN_ERROR);
			System.out.println(statistics[0] + "\t" + statistics[1] + "\t" + statistics[2] + "\t" + statistics[3]);
		}
	}

}
