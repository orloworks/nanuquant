/**
 * Rolling metrics for nanuquant.
 *
 * This module provides rolling (window-based) metrics.
 *
 * Note: Rolling metrics default to 365 periods per year to match QuantStats.
 * To use trading days, pass `periodsPerYear: 252` explicitly.
 */

import { RollingGreeks } from '../types';
import { getAnnualizationFactor, mean, std, sum } from './utils';
import { validateReturns, validateBenchmarkMatch } from './validation';

/**
 * Calculate rolling standard deviation.
 *
 * @param arr - Input array.
 * @param windowSize - Size of rolling window.
 * @param ddof - Delta degrees of freedom. Default: 1.
 * @returns Array of rolling standard deviations (first windowSize-1 values are NaN).
 */
function rollingStd(arr: number[], windowSize: number, ddof: number = 1): number[] {
  const result: number[] = new Array(arr.length).fill(NaN);

  for (let i = windowSize - 1; i < arr.length; i++) {
    const window = arr.slice(i - windowSize + 1, i + 1);
    result[i] = std(window, ddof);
  }

  return result;
}

/**
 * Calculate rolling mean.
 *
 * @param arr - Input array.
 * @param windowSize - Size of rolling window.
 * @returns Array of rolling means (first windowSize-1 values are NaN).
 */
function rollingMean(arr: number[], windowSize: number): number[] {
  const result: number[] = new Array(arr.length).fill(NaN);

  for (let i = windowSize - 1; i < arr.length; i++) {
    const window = arr.slice(i - windowSize + 1, i + 1);
    result[i] = mean(window);
  }

  return result;
}

/**
 * Calculate rolling sum.
 *
 * @param arr - Input array.
 * @param windowSize - Size of rolling window.
 * @returns Array of rolling sums (first windowSize-1 values are NaN).
 */
function rollingSum(arr: number[], windowSize: number): number[] {
  const result: number[] = new Array(arr.length).fill(NaN);

  for (let i = windowSize - 1; i < arr.length; i++) {
    const window = arr.slice(i - windowSize + 1, i + 1);
    result[i] = sum(window);
  }

  return result;
}

/**
 * Calculate rolling volatility.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.rollingPeriod - Window size. Default: 126 (6 months).
 * @param options.periodsPerYear - Periods per year for annualization. Default: 365.
 * @param options.annualize - Whether to annualize. Default: true.
 * @returns Array of rolling volatilities.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, -0.01, 0.02, ...];
 * rollingVolatility(returns, { rollingPeriod: 10 });
 * ```
 */
export function rollingVolatility(
  returns: number[],
  options: {
    rollingPeriod?: number;
    periodsPerYear?: number;
    annualize?: boolean;
  } = {}
): number[] {
  const validated = validateReturns(returns, { allowEmpty: true });
  const {
    rollingPeriod = 126,
    periodsPerYear = 365,
    annualize = true,
  } = options;

  if (validated.length < rollingPeriod) {
    return [];
  }

  const annFactor = getAnnualizationFactor(periodsPerYear);
  const rollStd = rollingStd(validated, rollingPeriod, 1);

  if (annualize) {
    return rollStd.map((v) => (Number.isNaN(v) ? NaN : v * Math.sqrt(annFactor)));
  }

  return rollStd;
}

/**
 * Calculate rolling Sharpe ratio.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.riskFreeRate - Annualized risk-free rate. Default: 0.
 * @param options.rollingPeriod - Window size. Default: 126.
 * @param options.periodsPerYear - Periods per year for annualization. Default: 365.
 * @param options.annualize - Whether to annualize. Default: true.
 * @returns Array of rolling Sharpe ratios.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, -0.01, 0.02, ...];
 * rollingSharpe(returns, { rollingPeriod: 10 });
 * ```
 */
export function rollingSharpe(
  returns: number[],
  options: {
    riskFreeRate?: number;
    rollingPeriod?: number;
    periodsPerYear?: number;
    annualize?: boolean;
  } = {}
): number[] {
  const validated = validateReturns(returns, { allowEmpty: true });
  const {
    riskFreeRate = 0.0,
    rollingPeriod = 126,
    periodsPerYear = 365,
    annualize = true,
  } = options;

  if (validated.length < rollingPeriod) {
    return [];
  }

  const annFactor = getAnnualizationFactor(periodsPerYear);

  // Convert annual risk-free to per-period
  const rfPerPeriod = riskFreeRate / annFactor;

  // Excess returns
  const excessReturns = validated.map((r) => r - rfPerPeriod);

  // Rolling mean of excess returns
  const rollMean = rollingMean(excessReturns, rollingPeriod);

  // Rolling std of original returns (like QuantStats)
  const rollStd = rollingStd(validated, rollingPeriod, 1);

  // Sharpe = mean_excess / std_returns
  const result: number[] = [];
  for (let i = 0; i < validated.length; i++) {
    if (Number.isNaN(rollMean[i]) || Number.isNaN(rollStd[i]) || rollStd[i] === 0) {
      result.push(NaN);
    } else if (annualize) {
      result.push((rollMean[i] / rollStd[i]) * Math.sqrt(annFactor));
    } else {
      result.push(rollMean[i] / rollStd[i]);
    }
  }

  return result;
}

/**
 * Calculate rolling Sortino ratio.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.riskFreeRate - Annualized risk-free rate. Default: 0.
 * @param options.rollingPeriod - Window size. Default: 126.
 * @param options.periodsPerYear - Periods per year for annualization. Default: 365.
 * @param options.annualize - Whether to annualize. Default: true.
 * @returns Array of rolling Sortino ratios.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, -0.01, 0.02, ...];
 * rollingSortino(returns, { rollingPeriod: 10 });
 * ```
 */
export function rollingSortino(
  returns: number[],
  options: {
    riskFreeRate?: number;
    rollingPeriod?: number;
    periodsPerYear?: number;
    annualize?: boolean;
  } = {}
): number[] {
  const validated = validateReturns(returns, { allowEmpty: true });
  const {
    riskFreeRate = 0.0,
    rollingPeriod = 126,
    periodsPerYear = 365,
    annualize = true,
  } = options;

  if (validated.length < rollingPeriod) {
    return [];
  }

  const annFactor = getAnnualizationFactor(periodsPerYear);

  // Convert annual risk-free to per-period
  const rfPerPeriod = riskFreeRate / annFactor;

  // Adjusted returns
  const adjusted = validated.map((r) => r - rfPerPeriod);

  // Rolling mean of adjusted returns
  const rollMean = rollingMean(adjusted, rollingPeriod);

  // For downside deviation: square negative values, sum in window, divide by n, sqrt
  const negativeSquared = adjusted.map((r) => (r < 0 ? r * r : 0));
  const rollNegSqSum = rollingSum(negativeSquared, rollingPeriod);

  const result: number[] = [];
  for (let i = 0; i < validated.length; i++) {
    if (Number.isNaN(rollMean[i]) || Number.isNaN(rollNegSqSum[i])) {
      result.push(NaN);
    } else {
      const downside = Math.sqrt(rollNegSqSum[i] / rollingPeriod);
      if (downside === 0) {
        result.push(rollMean[i] > 0 ? Infinity : NaN);
      } else if (annualize) {
        result.push((rollMean[i] / downside) * Math.sqrt(annFactor));
      } else {
        result.push(rollMean[i] / downside);
      }
    }
  }

  return result;
}

/**
 * Calculate rolling beta relative to benchmark.
 *
 * @param returns - Strategy period returns.
 * @param benchmark - Benchmark period returns.
 * @param options - Optional parameters.
 * @param options.rollingPeriod - Window size. Default: 126.
 * @returns Array of rolling betas.
 *
 * @example
 * ```ts
 * const returns = [0.02, -0.01, 0.03, 0.01, -0.02, ...];
 * const benchmark = [0.01, -0.02, 0.02, 0.01, -0.01, ...];
 * rollingBeta(returns, benchmark, { rollingPeriod: 10 });
 * ```
 */
export function rollingBeta(
  returns: number[],
  benchmark: number[],
  options: { rollingPeriod?: number } = {}
): number[] {
  const validatedReturns = validateReturns(returns, { allowEmpty: true });
  const validatedBenchmark = validateReturns(benchmark, { allowEmpty: true });
  validateBenchmarkMatch(validatedReturns, validatedBenchmark);

  const { rollingPeriod = 126 } = options;

  if (validatedReturns.length < rollingPeriod) {
    return [];
  }

  // Product of returns and benchmark
  const product = validatedReturns.map((r, i) => r * validatedBenchmark[i]);

  // Benchmark squared
  const benchSq = validatedBenchmark.map((b) => b * b);

  // Rolling means
  const rollMeanRet = rollingMean(validatedReturns, rollingPeriod);
  const rollMeanBench = rollingMean(validatedBenchmark, rollingPeriod);
  const rollMeanProduct = rollingMean(product, rollingPeriod);
  const rollMeanBenchSq = rollingMean(benchSq, rollingPeriod);

  // Beta = Cov / Var
  // Cov = E[XY] - E[X]E[Y]
  // Var = E[Y^2] - E[Y]^2
  const result: number[] = [];
  for (let i = 0; i < validatedReturns.length; i++) {
    if (
      Number.isNaN(rollMeanRet[i]) ||
      Number.isNaN(rollMeanBench[i]) ||
      Number.isNaN(rollMeanProduct[i]) ||
      Number.isNaN(rollMeanBenchSq[i])
    ) {
      result.push(NaN);
    } else {
      const cov = rollMeanProduct[i] - rollMeanRet[i] * rollMeanBench[i];
      const varBench = rollMeanBenchSq[i] - rollMeanBench[i] * rollMeanBench[i];
      if (varBench === 0) {
        result.push(NaN);
      } else {
        result.push(cov / varBench);
      }
    }
  }

  return result;
}

/**
 * Calculate rolling alpha and beta relative to benchmark.
 *
 * @param returns - Strategy period returns.
 * @param benchmark - Benchmark period returns.
 * @param options - Optional parameters.
 * @param options.rollingPeriod - Window size. Default: 126.
 * @param options.periodsPerYear - Periods per year for alpha annualization. Default: 365.
 * @returns Object with alpha and beta arrays.
 *
 * @example
 * ```ts
 * const returns = [0.02, -0.01, 0.03, 0.01, -0.02, ...];
 * const benchmark = [0.01, -0.02, 0.02, 0.01, -0.01, ...];
 * const { alpha, beta } = rollingGreeks(returns, benchmark);
 * ```
 */
export function rollingGreeks(
  returns: number[],
  benchmark: number[],
  options: { rollingPeriod?: number; periodsPerYear?: number } = {}
): RollingGreeks {
  const validatedReturns = validateReturns(returns, { allowEmpty: true });
  const validatedBenchmark = validateReturns(benchmark, { allowEmpty: true });
  validateBenchmarkMatch(validatedReturns, validatedBenchmark);

  const { rollingPeriod = 126, periodsPerYear = 365 } = options;

  if (validatedReturns.length < rollingPeriod) {
    return { alpha: [], beta: [] };
  }

  const annFactor = getAnnualizationFactor(periodsPerYear);

  // Product of returns and benchmark
  const product = validatedReturns.map((r, i) => r * validatedBenchmark[i]);

  // Benchmark squared
  const benchSq = validatedBenchmark.map((b) => b * b);

  // Rolling means
  const rollMeanRet = rollingMean(validatedReturns, rollingPeriod);
  const rollMeanBench = rollingMean(validatedBenchmark, rollingPeriod);
  const rollMeanProduct = rollingMean(product, rollingPeriod);
  const rollMeanBenchSq = rollingMean(benchSq, rollingPeriod);

  const alphaResult: number[] = [];
  const betaResult: number[] = [];

  for (let i = 0; i < validatedReturns.length; i++) {
    if (
      Number.isNaN(rollMeanRet[i]) ||
      Number.isNaN(rollMeanBench[i]) ||
      Number.isNaN(rollMeanProduct[i]) ||
      Number.isNaN(rollMeanBenchSq[i])
    ) {
      alphaResult.push(NaN);
      betaResult.push(NaN);
    } else {
      const cov = rollMeanProduct[i] - rollMeanRet[i] * rollMeanBench[i];
      const varBench = rollMeanBenchSq[i] - rollMeanBench[i] * rollMeanBench[i];

      if (varBench === 0) {
        alphaResult.push(NaN);
        betaResult.push(NaN);
      } else {
        const beta = cov / varBench;
        const alphaPeriod = rollMeanRet[i] - beta * rollMeanBench[i];
        const alpha = alphaPeriod * annFactor;

        alphaResult.push(alpha);
        betaResult.push(beta);
      }
    }
  }

  return { alpha: alphaResult, beta: betaResult };
}
