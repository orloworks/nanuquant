/**
 * Normal distribution functions for nanuquant.
 *
 * Provides approximations for normal distribution functions without external dependencies.
 */

/**
 * Standard normal cumulative distribution function (CDF).
 *
 * Uses Abramowitz and Stegun approximation (equation 7.1.26).
 * Maximum error: 1.5×10^-7
 *
 * @param x - Value to evaluate.
 * @returns P(Z <= x) for standard normal Z.
 */
export function normCdf(x: number): number {
  // Constants for the approximation
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  // Save the sign of x
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.SQRT2;

  // A&S formula 7.1.26
  const t = 1.0 / (1.0 + p * x);
  const y =
    1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return 0.5 * (1.0 + sign * y);
}

/**
 * Standard normal probability density function (PDF).
 *
 * @param x - Value to evaluate.
 * @returns PDF value at x.
 */
export function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/**
 * Standard normal quantile function (inverse CDF / percent point function).
 *
 * Uses Abramowitz and Stegun rational approximation (equation 26.2.23).
 * Accuracy: approximately 4.5×10^-4
 *
 * @param p - Probability value (0 < p < 1).
 * @param mean - Mean of the distribution. Default: 0.
 * @param std - Standard deviation of the distribution. Default: 1.
 * @returns The value x such that P(X <= x) = p.
 */
export function normPpf(p: number, mean: number = 0, std: number = 1): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return mean;

  // Use rational approximation for standard normal
  const sign = p < 0.5 ? -1 : 1;
  const p2 = p < 0.5 ? p : 1 - p;

  // Coefficients for rational approximation
  const c0 = 2.515517;
  const c1 = 0.802853;
  const c2 = 0.010328;
  const d1 = 1.432788;
  const d2 = 0.189269;
  const d3 = 0.001308;

  const t = Math.sqrt(-2 * Math.log(p2));
  const z =
    t -
    (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);

  return mean + sign * z * std;
}

/**
 * Higher accuracy normal quantile function using Acklam's algorithm.
 *
 * This provides better accuracy than the basic approximation.
 * Relative error: approximately 1.15×10^-9
 *
 * @param p - Probability value (0 < p < 1).
 * @param mean - Mean of the distribution. Default: 0.
 * @param std - Standard deviation of the distribution. Default: 1.
 * @returns The value x such that P(X <= x) = p.
 */
export function normPpfAccurate(
  p: number,
  mean: number = 0,
  std: number = 1
): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return mean;

  // Coefficients in rational approximations
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0,
  ];

  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];

  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
    -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0,
  ];

  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
    3.754408661907416e0,
  ];

  // Define break-points
  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q: number;
  let r: number;
  let z: number;

  if (p < pLow) {
    // Rational approximation for lower region
    q = Math.sqrt(-2 * Math.log(p));
    z =
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= pHigh) {
    // Rational approximation for central region
    q = p - 0.5;
    r = q * q;
    z =
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    // Rational approximation for upper region
    q = Math.sqrt(-2 * Math.log(1 - p));
    z =
      -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }

  return mean + z * std;
}
