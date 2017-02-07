package eu.veldsoft.digits.resilient;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationSIN;
import org.encog.mathutil.BoundMath;

class ActivationFadingSin implements ActivationFunction {
	private final ActivationSIN SIN = new ActivationSIN();
	private double period = 1.0D;

	public ActivationFadingSin(double period) {
		this.period = period;
	}

	public void activationFunction(double[] values, int start, int size) {
		for (int i = start; i < (start + size) && i < values.length; i++) {
			double x = values[i] / period;

			if (x < -Math.PI || x > Math.PI) {
				values[i] = BoundMath.sin(x) / Math.abs(x);
			} else {
				values[i] = BoundMath.sin(x);
			}
		}
	}

	public double derivativeFunction(double before, double after) {
		double x = before / period;

		if (x < -Math.PI || x > Math.PI) {
			return BoundMath.cos(x) / Math.abs(x) - BoundMath.sin(x) / (x * Math.abs(x));
		} else {
			return BoundMath.cos(x);
		}
	}

	public ActivationFunction clone() {
		return new ActivationFadingSin(period);
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