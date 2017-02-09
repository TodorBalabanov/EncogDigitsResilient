package eu.veldsoft.digits.resilient;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationSIN;
import org.encog.mathutil.BoundMath;

class ActivationExponentRegulatedSin implements ActivationFunction {
	private final ActivationSIN SIN = new ActivationSIN();

	private double period = 1.0D;

	static final double LOW = -0.99;

	static final double HIGH = +0.99;

	public ActivationExponentRegulatedSin(double period) {
		this.period = period;
	}

	public void activationFunction(double[] values, int start, int size) {
		for (int i = start; i < (start + size) && i < values.length; i++) {
			double x = values[i] / period;

			values[i] = Math.PI * BoundMath.sin(x) / BoundMath.exp(Math.abs(x));
		}
	}

	public double derivativeFunction(double before, double after) {
		double x = before / period;

		if (x == 0) {
			return Double.MAX_VALUE;
		}

		return Math.PI * BoundMath.exp(-Math.abs(x)) * (BoundMath.cos(x) * Math.abs(x) - x * BoundMath.sin(x))
				/ Math.abs(x);
	}

	public ActivationFunction clone() {
		return new ActivationExponentRegulatedSin(period);
	}

	public String getFactoryCode() {
		return null;
	}

	public String[] getParamNames() {
		return SIN.getParamNames();
	}

	public double[] getParams() {
		return SIN.getParams();
	}

	public boolean hasDerivative() {
		return true;
	}

	public void setParam(int index, double value) {
		SIN.setParam(index, value);
	}
}
