package eu.veldsoft.digits.resilient;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

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
	private static final int NUMBER_OF_EXPERIMENTS = 30;

	private static final int INPUT_SIZE = 256;

	private static final int OUTPUT_SIZE = 10;

	private static final int HIDDEN_SIZE = INPUT_SIZE + OUTPUT_SIZE;

	private static final NeuralDataSet ZERO_ONE_TRAINING = new BasicNeuralDataSet();

	private static final NeuralDataSet MINUS_PLUS_ONE_TRAINING = new BasicNeuralDataSet();

	static {
		/*
		 * It is used for time measurement calibration.
		 */
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

		PruneIncremental prune = new PruneIncremental(training, pattern, iterations, 9, 30,
				new ConsoleStatusReportable());

		prune.addHiddenLayer(hiddenMin, hiddenMax);

		prune.process();

		return prune.getBestNetwork();
	}

	private static List<Object> train1(String title, ActivationFunction activation, int optimalHiddenSize,
			NeuralDataSet training, double epsilon, long maxTrainingTime, long stopAtTime) {
		List<Object> result = new ArrayList<>();

		ActivationFunction activations[] = { activation, activation, activation };

		if (activation instanceof ActivationFadingSin1) {
			activations[0] = new ActivationFadingSin1(INPUT_SIZE);
			activations[1] = new ActivationFadingSin1(optimalHiddenSize);
			activations[2] = new ActivationFadingSin1(OUTPUT_SIZE);
		} else if (activation instanceof ActivationExponentRegulatedSin) {
			activations[0] = new ActivationExponentRegulatedSin(INPUT_SIZE);
			activations[1] = new ActivationExponentRegulatedSin(optimalHiddenSize);
			activations[2] = new ActivationExponentRegulatedSin(OUTPUT_SIZE);
		}

		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(activations[0], true, INPUT_SIZE));
		network.addLayer(new BasicLayer(activations[1], true, optimalHiddenSize));
		network.addLayer(new BasicLayer(activations[2], false, OUTPUT_SIZE));
		network.getStructure().finalizeStructure();
		network.reset();

		final Train train = new ResilientPropagation(network, training);

		/*
		 * Initial record.
		 */ {
			train.iteration();
			Object record[] = { title, Double.valueOf(train.getError()), Long.valueOf(stopAtTime), Long.valueOf(0) };
			result.add(record);
		}

		int epoch = 0;
		while (epsilon < train.getError()) {
			long start = System.currentTimeMillis();

			do {
				train.iteration();
				epoch++;
			} while ((System.currentTimeMillis() - start) < stopAtTime);

			Object record[] = { title, Double.valueOf(train.getError()),
					Long.valueOf((System.currentTimeMillis() - start)), Long.valueOf(epoch) };
			result.add(record);
		}

		return result;
	}

	private static List<Object> train2(String title, ActivationFunction activation, int optimalHiddenSize,
			NeuralDataSet training, int numberOfStops, long stopAtTime) {
		List<Object> result = new ArrayList<>();

		ActivationFunction activations[] = { activation, activation, activation };

		if (activation instanceof ActivationFadingSin1) {
			activations[0] = new ActivationFadingSin1(INPUT_SIZE);
			activations[1] = new ActivationFadingSin1(optimalHiddenSize);
			activations[2] = new ActivationFadingSin1(OUTPUT_SIZE);
		} else if (activation instanceof ActivationExponentRegulatedSin) {
			activations[0] = new ActivationExponentRegulatedSin(INPUT_SIZE);
			activations[1] = new ActivationExponentRegulatedSin(optimalHiddenSize);
			activations[2] = new ActivationExponentRegulatedSin(OUTPUT_SIZE);
		}

		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(activations[0], true, INPUT_SIZE));
		network.addLayer(new BasicLayer(activations[1], true, optimalHiddenSize));
		network.addLayer(new BasicLayer(activations[2], false, OUTPUT_SIZE));
		network.getStructure().finalizeStructure();
		network.reset();

		final Train train = new ResilientPropagation(network, training);

		/*
		 * Initial record.
		 */ {
			train.iteration();
			Object record[] = { title, Double.valueOf(train.getError()), Long.valueOf(stopAtTime), Long.valueOf(0) };
			result.add(record);
		}

		int epoch = 0;
		for (long g = 0; g < numberOfStops; g++) {
			long start = System.currentTimeMillis();

			do {
				train.iteration();
				epoch++;
			} while ((System.currentTimeMillis() - start) < stopAtTime);

			Object record[] = { title, Double.valueOf(train.getError()),
					Long.valueOf((System.currentTimeMillis() - start)), Long.valueOf(epoch) };
			result.add(record);
		}

		return result;
	}

	private static void prune() {
		final int MIN_HIDDEN = 1;
		final int MAX_HIDDEN = INPUT_SIZE + OUTPUT_SIZE;
		final int PRUNE_ITERATIONS = 900;

		BasicNetwork net = null;
		net = prune("Fading Sine", new ActivationFadingSin1(1), MINUS_PLUS_ONE_TRAINING, PRUNE_ITERATIONS, MIN_HIDDEN,
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
	}

	private static void train1() {
		final double TARGET_ANN_ERROR = 0.1;
		final long MAX_TRAINING_TIME = 10 * 60 * 1000;
		final long SINGLE_MEASUREMENT_MILLISECONDS = 1000 * 1;

		List<Object> statistics = null;

		statistics = train1("Fading Sine", new ActivationFadingSin1(1), 85, MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR,
				MAX_TRAINING_TIME, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train1("Exponent Regulated Sine", new ActivationExponentRegulatedSin(1), 106,
				MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR, MAX_TRAINING_TIME, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train1("Sigmoid", new ActivationSigmoid(), 155, ZERO_ONE_TRAINING, TARGET_ANN_ERROR,
				MAX_TRAINING_TIME, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train1("Bipolar Sigmoid", new ActivationBipolarSteepenedSigmoid(), 188, MINUS_PLUS_ONE_TRAINING,
				TARGET_ANN_ERROR, MAX_TRAINING_TIME, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train1("Logarithm", new ActivationLOG(), 11, MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR,
				MAX_TRAINING_TIME, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train1("Hyperbolic Tangent", new ActivationTANH(), 44, MINUS_PLUS_ONE_TRAINING, TARGET_ANN_ERROR,
				MAX_TRAINING_TIME, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train1("Elliott Symmetric", new ActivationElliottSymmetric(), 162, MINUS_PLUS_ONE_TRAINING,
				TARGET_ANN_ERROR, MAX_TRAINING_TIME, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
	}

	private static void train2() {
		final int NUMBER_OF_MEASUREMENTS = 60;
		final long SINGLE_MEASUREMENT_MILLISECONDS = 1000;

		List<Object> statistics = null;

		statistics = train2("Fading Sine", new ActivationFadingSin1(1), 85, MINUS_PLUS_ONE_TRAINING,
				NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Exponent Regulated Sine", new ActivationExponentRegulatedSin(1), 106,
				MINUS_PLUS_ONE_TRAINING, NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Sigmoid", new ActivationSigmoid(), 155, ZERO_ONE_TRAINING, NUMBER_OF_MEASUREMENTS,
				SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Bipolar Sigmoid", new ActivationBipolarSteepenedSigmoid(), 188, MINUS_PLUS_ONE_TRAINING,
				NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Logarithm", new ActivationLOG(), 11, MINUS_PLUS_ONE_TRAINING, NUMBER_OF_MEASUREMENTS,
				SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Hyperbolic Tangent", new ActivationTANH(), 44, MINUS_PLUS_ONE_TRAINING,
				NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Elliott Symmetric", new ActivationElliottSymmetric(), 162, MINUS_PLUS_ONE_TRAINING,
				NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
	}

	private static void train3() {
		final int NUMBER_OF_MEASUREMENTS = 60;
		final long SINGLE_MEASUREMENT_MILLISECONDS = 1000;

		List<Object> statistics = null;

		statistics = train2("Fading Sine", new ActivationFadingSin1(1), 64, MINUS_PLUS_ONE_TRAINING,
				NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Exponent Regulated Sine", new ActivationExponentRegulatedSin(1), 64,
				MINUS_PLUS_ONE_TRAINING, NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Sigmoid", new ActivationSigmoid(), 64, ZERO_ONE_TRAINING, NUMBER_OF_MEASUREMENTS,
				SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Bipolar Sigmoid", new ActivationBipolarSteepenedSigmoid(), 64, MINUS_PLUS_ONE_TRAINING,
				NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Logarithm", new ActivationLOG(), 64, MINUS_PLUS_ONE_TRAINING, NUMBER_OF_MEASUREMENTS,
				SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Hyperbolic Tangent", new ActivationTANH(), 64, MINUS_PLUS_ONE_TRAINING,
				NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Elliott Symmetric", new ActivationElliottSymmetric(), 64, MINUS_PLUS_ONE_TRAINING,
				NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
	}

	private static void train4() {
		final int NUMBER_OF_MEASUREMENTS = 600;
		final long SINGLE_MEASUREMENT_MILLISECONDS = 1000;

		List<Object> statistics = null;

		statistics = train2("Fading Sine Original Derivative", new ActivationFadingSin1(1), 64, MINUS_PLUS_ONE_TRAINING,
				NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
		statistics = train2("Fading Sine Alternative Derivative", new ActivationFadingSin2(1), 64,
				MINUS_PLUS_ONE_TRAINING, NUMBER_OF_MEASUREMENTS, SINGLE_MEASUREMENT_MILLISECONDS);
		System.out.println(Arrays.deepToString((Object[]) statistics.toArray()));
	}

	private static void train5() {
	}

	public static void main(final String args[]) {
		// prune();
		// train1();
		// train2();
		// train3();
		// train4();
		train5();
	}
}
