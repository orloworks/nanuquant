/**
 * Correlation metrics for nanuquant.
 *
 * This module provides correlation-based metrics including conditional correlations.
 */

import { mean, std, sum } from './utils';
import { validateReturns, validateBenchmarkMatch } from './validation';

/**
 * Calculate Pearson correlation coefficient.
 *
 * @param x - First array.
 * @param y - Second array.
 * @returns Pearson correlation coefficient (-1 to 1).
 *
 * @example
 * ```ts
 * const x = [0.02, -0.01, 0.03, 0.01, -0.02];
 * const y = [0.01, -0.02, 0.02, 0.01, -0.01];
 * correlation(x, y); // ~0.77
 * ```
 */
export function correlation(x: number[], y: number[]): number {
  const validatedX = validateReturns(x, { allowEmpty: true });
  const validatedY = validateReturns(y, { allowEmpty: true });
  validateBenchmarkMatch(validatedX, validatedY);

  if (validatedX.length < 2) {
    return 0.0;
  }

  const n = validatedX.length;
  const meanX = mean(validatedX);
  const meanY = mean(validatedY);

  // Sample covariance (N-1 denominator)
  const covSum = sum(
    validatedX.map((xi, i) => (xi - meanX) * (validatedY[i] - meanY))
  );
  const cov = covSum / (n - 1);

  // Sample standard deviations
  const stdX = std(validatedX, 1);
  const stdY = std(validatedY, 1);

  if (stdX === 0 || stdY === 0) {
    return 0.0;
  }

  return cov / (stdX * stdY);
}

/**
 * Calculate sample covariance.
 *
 * @param x - First array.
 * @param y - Second array.
 * @param ddof - Delta degrees of freedom. Default: 1 (sample covariance).
 * @returns Sample covariance.
 *
 * @example
 * ```ts
 * const x = [0.02, -0.01, 0.03, 0.01, -0.02];
 * const y = [0.01, -0.02, 0.02, 0.01, -0.01];
 * covariance(x, y); // ~0.000175
 * ```
 */
export function covariance(x: number[], y: number[], ddof: number = 1): number {
  const validatedX = validateReturns(x, { allowEmpty: true });
  const validatedY = validateReturns(y, { allowEmpty: true });
  validateBenchmarkMatch(validatedX, validatedY);

  if (validatedX.length <= ddof) {
    return 0.0;
  }

  const n = validatedX.length;
  const meanX = mean(validatedX);
  const meanY = mean(validatedY);

  const covSum = sum(
    validatedX.map((xi, i) => (xi - meanX) * (validatedY[i] - meanY))
  );

  return covSum / (n - ddof);
}

/**
 * Calculate correlation during downside regimes.
 *
 * Measures correlation specifically during periods when the first series
 * experiences returns below the threshold.
 *
 * @param returns1 - First return series (used to define downside regime).
 * @param returns2 - Second return series.
 * @param options - Optional parameters.
 * @param options.threshold - Return threshold defining "downside". Default: 0.
 * @returns Pearson correlation during downside periods (-1 to 1), or NaN if insufficient data.
 *
 * @example
 * ```ts
 * const r1 = [0.02, -0.03, -0.01, 0.01, -0.02];
 * const r2 = [0.01, -0.02, 0.02, 0.01, -0.01];
 * downsideCorrelation(r1, r2); // correlation when r1 < 0
 * ```
 */
export function downsideCorrelation(
  returns1: number[],
  returns2: number[],
  options: { threshold?: number } = {}
): number {
  const validated1 = validateReturns(returns1, { allowEmpty: true });
  const validated2 = validateReturns(returns2, { allowEmpty: true });
  validateBenchmarkMatch(validated1, validated2);

  const { threshold = 0.0 } = options;

  // Filter to downside periods (where returns1 < threshold)
  const indices: number[] = [];
  for (let i = 0; i < validated1.length; i++) {
    if (validated1[i] < threshold) {
      indices.push(i);
    }
  }

  if (indices.length < 2) {
    return NaN;
  }

  const r1Down = indices.map((i) => validated1[i]);
  const r2Down = indices.map((i) => validated2[i]);

  return correlation(r1Down, r2Down);
}

/**
 * Calculate correlation during upside regimes.
 *
 * Measures correlation specifically during periods when the first series
 * experiences returns above the threshold.
 *
 * @param returns1 - First return series (used to define upside regime).
 * @param returns2 - Second return series.
 * @param options - Optional parameters.
 * @param options.threshold - Return threshold defining "upside". Default: 0.
 * @returns Pearson correlation during upside periods (-1 to 1), or NaN if insufficient data.
 *
 * @example
 * ```ts
 * const r1 = [0.02, -0.03, 0.01, 0.03, -0.02];
 * const r2 = [0.01, -0.02, 0.02, 0.01, -0.01];
 * upsideCorrelation(r1, r2); // correlation when r1 > 0
 * ```
 */
export function upsideCorrelation(
  returns1: number[],
  returns2: number[],
  options: { threshold?: number } = {}
): number {
  const validated1 = validateReturns(returns1, { allowEmpty: true });
  const validated2 = validateReturns(returns2, { allowEmpty: true });
  validateBenchmarkMatch(validated1, validated2);

  const { threshold = 0.0 } = options;

  // Filter to upside periods (where returns1 > threshold)
  const indices: number[] = [];
  for (let i = 0; i < validated1.length; i++) {
    if (validated1[i] > threshold) {
      indices.push(i);
    }
  }

  if (indices.length < 2) {
    return NaN;
  }

  const r1Up = indices.map((i) => validated1[i]);
  const r2Up = indices.map((i) => validated2[i]);

  return correlation(r1Up, r2Up);
}

/**
 * Calculate rolling correlation.
 *
 * @param x - First array.
 * @param y - Second array.
 * @param windowSize - Size of the rolling window.
 * @returns Array of rolling correlations (first windowSize-1 values are NaN).
 *
 * @example
 * ```ts
 * const x = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02];
 * const y = [0.005, 0.01, -0.02, 0.02, 0.005, -0.01, 0.01];
 * rollingCorrelation(x, y, 5);
 * ```
 */
export function rollingCorrelation(
  x: number[],
  y: number[],
  windowSize: number
): number[] {
  const validatedX = validateReturns(x, { allowEmpty: true });
  const validatedY = validateReturns(y, { allowEmpty: true });
  validateBenchmarkMatch(validatedX, validatedY);

  const result: number[] = new Array(validatedX.length).fill(NaN);

  for (let i = windowSize - 1; i < validatedX.length; i++) {
    const windowX = validatedX.slice(i - windowSize + 1, i + 1);
    const windowY = validatedY.slice(i - windowSize + 1, i + 1);
    result[i] = correlation(windowX, windowY);
  }

  return result;
}
