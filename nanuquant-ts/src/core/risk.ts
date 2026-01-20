/**
 * Risk metrics for nanuquant.
 *
 * This module provides risk-based metrics that match Python NanuQuant output.
 */

import { getConfig } from '../config';
import { normPpfAccurate } from '../stats/normal';
import {
  getAnnualizationFactor,
  mean,
  std,
  sum,
  cumProd,
  cumMax,
  min as arrayMin,
  filter,
} from './utils';
import { validateReturns } from './validation';

/**
 * Calculate return volatility (standard deviation).
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.periodsPerYear - Number of periods per year for annualization.
 * @param options.annualize - If true (default), annualize the volatility.
 * @returns Annualized (or raw) volatility of returns.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, -0.01, 0.02];
 * volatility(returns, { periodsPerYear: 252 }); // ~0.248
 * ```
 */
export function volatility(
  returns: number[],
  options: { periodsPerYear?: number; annualize?: boolean } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const config = getConfig();
  const { periodsPerYear, annualize = true } = options;

  // Use sample std dev (ddof=1)
  const stdDev = std(validated, config.ddof);

  if (annualize) {
    const annFactor = getAnnualizationFactor(
      periodsPerYear ?? config.periodsPerYear
    );
    return stdDev * Math.sqrt(annFactor);
  }

  return stdDev;
}

/**
 * Calculate Value at Risk (VaR) using parametric method.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.sigma - Sigma multiplier for standard deviation. Default: 1.0.
 * @param options.confidence - Confidence level (e.g., 0.95 for 95%). Default: 0.95.
 * @returns VaR as a negative number representing the return at the given confidence.
 *
 * @example
 * ```ts
 * const returns = [-0.05, -0.02, 0.01, 0.03, -0.01];
 * varMetric(returns, { confidence: 0.95 }); // ~ -0.058
 * ```
 */
export function varMetric(
  returns: number[],
  options: { sigma?: number; confidence?: number } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const { sigma = 1.0, confidence = 0.95 } = options;

  const meanRet = mean(validated);
  const stdRet = std(validated, 1); // Sample std

  if (stdRet === 0) {
    return meanRet;
  }

  // Use parametric VaR: norm.ppf(1 - confidence, mean, sigma * std)
  return normPpfAccurate(1 - confidence, meanRet, sigma * stdRet);
}

// Alias for cleaner exports (var is a reserved word in JS)
export { varMetric as var };

/**
 * Calculate Conditional Value at Risk (Expected Shortfall).
 *
 * CVaR is the expected loss given that loss exceeds VaR.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.sigma - Sigma multiplier for VaR calculation. Default: 1.0.
 * @param options.confidence - Confidence level. Default: 0.95.
 * @returns CVaR as a negative number representing expected return in tail.
 *
 * @example
 * ```ts
 * const returns = [-0.10, -0.05, -0.02, 0.01, 0.03];
 * cvar(returns, { confidence: 0.95 }); // -0.10
 * ```
 */
export function cvar(
  returns: number[],
  options: { sigma?: number; confidence?: number } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const { sigma = 1.0, confidence = 0.95 } = options;

  // Get the VaR threshold using parametric method
  const varThreshold = varMetric(validated, { sigma, confidence });

  // Calculate expected loss in the tail (values < VaR)
  const tailLosses = filter(validated, (r) => r < varThreshold);

  if (tailLosses.length === 0) {
    // If no values in tail, return VaR
    return varThreshold;
  }

  return mean(tailLosses);
}

/**
 * Convert returns to drawdown series.
 *
 * @param returns - Period returns.
 * @returns Drawdown series (always negative or zero values).
 *
 * @example
 * ```ts
 * const returns = [0.10, -0.05, 0.02, -0.08];
 * toDrawdownSeries(returns);
 * // [0, -0.0454..., -0.0272..., -0.1054...]
 * ```
 */
export function toDrawdownSeries(returns: number[]): number[] {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return [];
  }

  // Calculate cumulative wealth (1 + cumulative return)
  const cumulative = cumProd(validated);

  // Running maximum
  const runningMax = cumMax(cumulative);

  // Drawdown as percentage from peak
  const drawdown: number[] = [];
  for (let i = 0; i < cumulative.length; i++) {
    drawdown.push((cumulative[i] - runningMax[i]) / runningMax[i]);
  }

  return drawdown;
}

/**
 * Calculate maximum drawdown.
 *
 * @param returns - Period returns.
 * @returns Maximum drawdown as a negative number (e.g., -0.30 means 30% drawdown).
 *
 * @example
 * ```ts
 * const returns = [0.10, -0.05, 0.02, -0.15, 0.05];
 * maxDrawdown(returns); // ~ -0.134
 * ```
 */
export function maxDrawdown(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const ddSeries = toDrawdownSeries(validated);
  if (ddSeries.length === 0) {
    return 0.0;
  }

  return arrayMin(ddSeries);
}

/**
 * Calculate the Ulcer Index (drawdown severity measure).
 *
 * @param returns - Period returns.
 * @returns Ulcer Index (always positive).
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, -0.03, 0.02];
 * ulcerIndex(returns); // ~ 0.015
 * ```
 */
export function ulcerIndex(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const n = validated.length;
  if (n <= 1) {
    return 0.0;
  }

  const ddSeries = toDrawdownSeries(validated);

  // Square the drawdowns and sum
  const ddSquaredSum = sum(ddSeries.map((d) => d * d));

  // Use (n - 1) denominator to match QuantStats
  return Math.sqrt(ddSquaredSum / (n - 1));
}

/**
 * Calculate downside deviation (semi-deviation below MAR).
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.mar - Minimum Acceptable Return (per period). Default: 0.
 * @param options.periodsPerYear - Periods per year for annualization.
 * @param options.annualize - Whether to annualize the result. Default: true.
 * @returns Downside deviation.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, -0.03, 0.02];
 * downsideDeviation(returns, { mar: 0.0 }); // ~ 0.18
 * ```
 */
export function downsideDeviation(
  returns: number[],
  options: { mar?: number; periodsPerYear?: number; annualize?: boolean } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const config = getConfig();
  const { mar = config.mar, periodsPerYear, annualize = true } = options;

  const n = validated.length;

  // Get returns below target
  const excess = validated.map((r) => r - mar);
  const negativeReturns = filter(excess, (r) => r < 0);

  if (negativeReturns.length === 0) {
    return 0.0;
  }

  // Sum of squared negative returns divided by total count
  const squaredSum = sum(negativeReturns.map((r) => r * r));
  const dd = Math.sqrt(squaredSum / n);

  if (annualize) {
    const annFactor = getAnnualizationFactor(
      periodsPerYear ?? config.periodsPerYear
    );
    return dd * Math.sqrt(annFactor);
  }

  return dd;
}
