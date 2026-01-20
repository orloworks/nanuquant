/**
 * Return metrics for nanuquant.
 *
 * This module provides return-based metrics that match Python NanuQuant output.
 */

import { getConfig } from '../config';
import {
  getAnnualizationFactor,
  safeDivide,
  mean,
  sum,
  filter,
  count,
  max as arrayMax,
  min as arrayMin,
} from './utils';
import { validateReturns } from './validation';

/**
 * Calculate total compounded return.
 *
 * @param returns - Period returns (not prices).
 * @returns Total compounded return. For example, 0.50 means 50% total return.
 *
 * @example
 * ```ts
 * const returns = [0.10, 0.05, -0.02];
 * comp(returns); // 0.1319
 * ```
 */
export function comp(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  // Product of (1 + r) for all r, minus 1
  const product = validated.reduce((acc, r) => acc * (1 + r), 1);
  return product - 1;
}

/**
 * Calculate Compound Annual Growth Rate.
 *
 * @param returns - Period returns (not prices).
 * @param options - Optional parameters.
 * @param options.periodsPerYear - Number of periods per year for annualization.
 * @returns Compound Annual Growth Rate. For example, 0.15 means 15% annual return.
 *
 * @example
 * ```ts
 * const returns = Array(252).fill(0.01); // 1% daily for a year
 * cagr(returns, { periodsPerYear: 252 }); // ~1134.6% annualized
 * ```
 */
export function cagr(
  returns: number[],
  options: { periodsPerYear?: number } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const config = getConfig();
  const annFactor = getAnnualizationFactor(
    options.periodsPerYear ?? config.periodsPerYear
  );

  const totalReturn = comp(validated);

  // Handle negative total returns that would cause issues with fractional exponents
  if (totalReturn <= -1) {
    return -Infinity;
  }

  const years = validated.length / annFactor;
  if (years === 0) {
    return 0.0;
  }

  return Math.pow(1 + totalReturn, 1 / years) - 1;
}

/**
 * Calculate average (arithmetic mean) return.
 *
 * @param returns - Period returns.
 * @returns Arithmetic mean of returns.
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, -0.01, 0.03];
 * avgReturn(returns); // 0.0125
 * ```
 */
export function avgReturn(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  return mean(validated);
}

/**
 * Calculate average winning (positive) return.
 *
 * @param returns - Period returns.
 * @returns Average of positive returns. Returns 0.0 if no positive returns.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.03, -0.01];
 * avgWin(returns); // 0.02
 * ```
 */
export function avgWin(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const wins = filter(validated, (r) => r > 0);
  if (wins.length === 0) {
    return 0.0;
  }

  return mean(wins);
}

/**
 * Calculate average losing (negative) return.
 *
 * @param returns - Period returns.
 * @returns Average of negative returns. Returns 0.0 if no negative returns.
 *         Note: This returns a negative value.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.03, -0.01];
 * avgLoss(returns); // -0.015
 * ```
 */
export function avgLoss(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const losses = filter(validated, (r) => r < 0);
  if (losses.length === 0) {
    return 0.0;
  }

  return mean(losses);
}

/**
 * Get the best (highest) single-period return.
 *
 * @param returns - Period returns.
 * @returns Maximum return in the series.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.05, -0.01];
 * best(returns); // 0.05
 * ```
 */
export function best(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  return arrayMax(validated);
}

/**
 * Get the worst (lowest) single-period return.
 *
 * @param returns - Period returns.
 * @returns Minimum return in the series.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.05, -0.01];
 * worst(returns); // -0.02
 * ```
 */
export function worst(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  return arrayMin(validated);
}

/**
 * Calculate percentage of positive returns.
 *
 * @param returns - Period returns.
 * @returns Fraction of returns that are positive (0 to 1).
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.03, -0.01];
 * winRate(returns); // 0.5
 * ```
 */
export function winRate(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const nWins = count(validated, (r) => r > 0);
  return safeDivide(nWins, validated.length, 0.0);
}

/**
 * Calculate payoff ratio (average win / average loss).
 *
 * @param returns - Period returns.
 * @returns Ratio of average win to absolute average loss.
 *          Returns 0.0 if no wins or losses.
 *
 * @example
 * ```ts
 * const returns = [0.02, -0.01, 0.03, -0.01];
 * payoffRatio(returns); // 2.5
 * ```
 */
export function payoffRatio(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const avgW = avgWin(validated);
  const avgL = avgLoss(validated);

  // avgLoss returns negative, take absolute value
  if (avgL === 0) {
    return avgW > 0 ? Infinity : 0.0;
  }

  return safeDivide(avgW, Math.abs(avgL), 0.0);
}

/**
 * Calculate profit factor (sum of wins / sum of losses).
 *
 * @param returns - Period returns.
 * @returns Ratio of gross profits to gross losses.
 *          Returns inf if no losses, 0.0 if no wins.
 *
 * @example
 * ```ts
 * const returns = [0.02, -0.01, 0.03, -0.01];
 * profitFactor(returns); // 2.5
 * ```
 */
export function profitFactor(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const wins = filter(validated, (r) => r > 0);
  const losses = filter(validated, (r) => r < 0);

  const sumWins = wins.length > 0 ? sum(wins) : 0.0;
  const sumLosses = losses.length > 0 ? Math.abs(sum(losses)) : 0.0;

  if (sumLosses === 0) {
    return sumWins > 0 ? Infinity : 0.0;
  }

  return safeDivide(sumWins, sumLosses, 0.0);
}

/**
 * Calculate maximum consecutive winning periods.
 *
 * @param returns - Period returns.
 * @returns Maximum number of consecutive positive returns.
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, 0.03, -0.01, 0.02, 0.01];
 * consecutiveWins(returns); // 3
 * ```
 */
export function consecutiveWins(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0;
  }

  let maxConsecutive = 0;
  let currentConsecutive = 0;

  for (const r of validated) {
    if (r > 0) {
      currentConsecutive++;
      maxConsecutive = Math.max(maxConsecutive, currentConsecutive);
    } else {
      currentConsecutive = 0;
    }
  }

  return maxConsecutive;
}

/**
 * Calculate maximum consecutive losing periods.
 *
 * @param returns - Period returns.
 * @returns Maximum number of consecutive negative returns.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, -0.03, -0.01, 0.02, 0.01];
 * consecutiveLosses(returns); // 3
 * ```
 */
export function consecutiveLosses(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0;
  }

  let maxConsecutive = 0;
  let currentConsecutive = 0;

  for (const r of validated) {
    if (r < 0) {
      currentConsecutive++;
      maxConsecutive = Math.max(maxConsecutive, currentConsecutive);
    } else {
      currentConsecutive = 0;
    }
  }

  return maxConsecutive;
}
