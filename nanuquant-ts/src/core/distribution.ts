/**
 * Distribution metrics for nanuquant.
 *
 * This module provides statistical distribution metrics.
 * Note: jarque_bera and shapiro_wilk tests are NOT included as they require
 * statistical distribution functions (chi-squared, etc.) that need scipy.
 */

import { mean, std, sum, filter, quantile } from './utils';
import { validateReturns } from './validation';

/**
 * Calculate skewness of returns (Fisher-Pearson, bias corrected).
 *
 * @param returns - Period returns.
 * @returns Skewness. Negative indicates left tail, positive indicates right tail.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.03, -0.01, 0.05];
 * skewness(returns); // ~0.53
 * ```
 */
export function skewness(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length < 3) {
    return 0.0;
  }

  const n = validated.length;
  const m = mean(validated);
  const s = std(validated, 1); // sample std

  if (s === 0) {
    return 0.0;
  }

  // Calculate third moment
  const m3 = sum(validated.map((x) => Math.pow((x - m) / s, 3))) / n;

  // Bias correction factor (Fisher-Pearson)
  const correction = Math.sqrt((n * (n - 1))) / (n - 2);

  return m3 * correction;
}

/**
 * Calculate excess kurtosis of returns (Fisher, bias corrected).
 *
 * @param returns - Period returns.
 * @returns Excess kurtosis. Values > 0 indicate fat tails.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.03, -0.01, 0.05];
 * kurtosis(returns); // ~ -1.38
 * ```
 */
export function kurtosis(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length < 4) {
    return 0.0;
  }

  const n = validated.length;
  const m = mean(validated);
  const s = std(validated, 1); // sample std

  if (s === 0) {
    return 0.0;
  }

  // Calculate fourth moment
  const m4 = sum(validated.map((x) => Math.pow((x - m) / s, 4))) / n;

  // Bias correction factor (Fisher)
  const g2 = m4 - 3; // excess kurtosis (raw)

  // Apply bias correction
  const correction =
    ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6);

  return correction;
}

/**
 * Calculate expected return (mean).
 *
 * @param returns - Period returns.
 * @returns Expected (mean) return.
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, -0.01, 0.03];
 * expectedReturn(returns); // 0.0125
 * ```
 */
export function expectedReturn(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  return mean(validated);
}

/**
 * Calculate geometric mean of returns.
 *
 * @param returns - Period returns.
 * @returns Geometric mean return per period.
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, -0.01, 0.03];
 * geometricMean(returns); // ~0.0123
 * ```
 */
export function geometricMean(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const n = validated.length;

  // Calculate product of (1 + returns)
  const product = validated.reduce((acc, r) => acc * (1 + r), 1);

  if (product <= 0) {
    return 0.0;
  }

  return Math.pow(product, 1.0 / n) - 1;
}

/**
 * Calculate ratio of outlier wins to mean win.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.q - Quantile threshold for outliers. Default: 0.99.
 * @returns Ratio of outlier wins (above quantile) to average win.
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, 0.03, 0.10, -0.01];
 * outlierWinRatio(returns, { q: 0.95 }); // ~2.5
 * ```
 */
export function outlierWinRatio(
  returns: number[],
  options: { q?: number } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const { q = 0.99 } = options;

  const wins = filter(validated, (r) => r > 0);
  if (wins.length === 0) {
    return 0.0;
  }

  const meanWin = mean(wins);
  if (meanWin === 0) {
    return 0.0;
  }

  const threshold = quantile(wins, q);
  const outliers = filter(wins, (r) => r >= threshold);

  if (outliers.length === 0) {
    return 0.0;
  }

  const meanOutlier = mean(outliers);
  return meanOutlier / meanWin;
}

/**
 * Calculate ratio of outlier losses to mean loss.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.q - Quantile threshold for outliers. Default: 0.99.
 * @returns Ratio of outlier losses (absolute) to absolute average loss.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, -0.03, -0.15, 0.02];
 * outlierLossRatio(returns, { q: 0.95 }); // ~2.0
 * ```
 */
export function outlierLossRatio(
  returns: number[],
  options: { q?: number } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const { q = 0.99 } = options;

  const losses = filter(validated, (r) => r < 0).map(Math.abs);
  if (losses.length === 0) {
    return 0.0;
  }

  const meanLoss = mean(losses);
  if (meanLoss === 0) {
    return 0.0;
  }

  const threshold = quantile(losses, q);
  const outliers = filter(losses, (r) => r >= threshold);

  if (outliers.length === 0) {
    return 0.0;
  }

  const meanOutlier = mean(outliers);
  return meanOutlier / meanLoss;
}

/**
 * Identify returns that exceed the specified quantile threshold.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.q - Quantile threshold. Default: 0.95.
 * @returns Array containing only the outlier returns (values above the quantile).
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.10, 0.02, -0.01, 0.15, 0.03];
 * outliers(returns, { q: 0.90 }); // [0.15]
 * ```
 */
export function outliers(
  returns: number[],
  options: { q?: number } = {}
): number[] {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return [];
  }

  const { q = 0.95 } = options;

  const threshold = quantile(validated, q);
  return filter(validated, (r) => r > threshold);
}

/**
 * Remove returns that exceed the specified quantile threshold.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.q - Quantile threshold. Default: 0.95.
 * @returns Array with outlier returns removed.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.10, 0.02, -0.01, 0.15, 0.03];
 * removeOutliers(returns, { q: 0.90 }); // 6 elements
 * ```
 */
export function removeOutliers(
  returns: number[],
  options: { q?: number } = {}
): number[] {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return [];
  }

  const { q = 0.95 } = options;

  const threshold = quantile(validated, q);
  return filter(validated, (r) => r <= threshold);
}

/**
 * Identify outliers using the IQR (Interquartile Range) method.
 *
 * @param returns - Period returns.
 * @returns Array containing only the outlier returns.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.10, 0.02, -0.01, -0.15, 0.03];
 * outliersIqr(returns);
 * ```
 */
export function outliersIqr(returns: number[]): number[] {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return [];
  }

  const q1 = quantile(validated, 0.25);
  const q3 = quantile(validated, 0.75);

  const iqr = q3 - q1;
  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;

  return filter(validated, (r) => r < lowerBound || r > upperBound);
}

/**
 * Remove outliers using the IQR (Interquartile Range) method.
 *
 * @param returns - Period returns.
 * @returns Array with outliers removed.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.10, 0.02, -0.01, -0.15, 0.03];
 * removeOutliersIqr(returns);
 * ```
 */
export function removeOutliersIqr(returns: number[]): number[] {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return [];
  }

  const q1 = quantile(validated, 0.25);
  const q3 = quantile(validated, 0.75);

  const iqr = q3 - q1;
  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;

  return filter(validated, (r) => r >= lowerBound && r <= upperBound);
}
