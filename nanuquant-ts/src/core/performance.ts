/**
 * Performance ratio metrics for nanuquant.
 *
 * This module provides risk-adjusted performance metrics that match Python NanuQuant output.
 */

import { getConfig } from '../config';
import { Greeks } from '../types';
import { comp, cagr, profitFactor } from './returns';
import { maxDrawdown, ulcerIndex } from './risk';
import {
  getAnnualizationFactor,
  safeDivide,
  mean,
  std,
  sum,
  filter,
  quantile,
} from './utils';
import { validateReturns, validateBenchmarkMatch } from './validation';

/**
 * Calculate the Sharpe ratio.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.riskFreeRate - Annualized risk-free rate. Default: 0.
 * @param options.periodsPerYear - Periods per year for annualization.
 * @returns Sharpe ratio (excess return / volatility).
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, -0.01, 0.015, 0.005];
 * sharpe(returns, { riskFreeRate: 0.0, periodsPerYear: 252 }); // ~2.54
 * ```
 */
export function sharpe(
  returns: number[],
  options: { riskFreeRate?: number; periodsPerYear?: number } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const config = getConfig();
  const { riskFreeRate = config.riskFreeRate, periodsPerYear } = options;
  const annFactor = getAnnualizationFactor(
    periodsPerYear ?? config.periodsPerYear
  );

  // Convert annual risk-free to per-period
  const rfPerPeriod = riskFreeRate / annFactor;

  // Excess returns
  const excessReturns = validated.map((r) => r - rfPerPeriod);

  const meanExcess = mean(excessReturns);
  const stdReturns = std(validated, config.ddof);

  if (stdReturns === 0) {
    return 0.0;
  }

  // Annualized Sharpe
  return (meanExcess / stdReturns) * Math.sqrt(annFactor);
}

/**
 * Calculate the Sortino ratio.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.riskFreeRate - Annualized risk-free rate. Default: 0.
 * @param options.periodsPerYear - Periods per year for annualization.
 * @returns Sortino ratio (mean return / downside deviation).
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, -0.01, 0.015, 0.005];
 * sortino(returns, { riskFreeRate: 0.0, periodsPerYear: 252 }); // ~3.60
 * ```
 */
export function sortino(
  returns: number[],
  options: { riskFreeRate?: number; periodsPerYear?: number } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const config = getConfig();
  const { riskFreeRate = config.riskFreeRate, periodsPerYear } = options;
  const annFactor = getAnnualizationFactor(
    periodsPerYear ?? config.periodsPerYear
  );

  // Convert annual risk-free to per-period
  const rfPerPeriod = riskFreeRate / annFactor;

  // Adjust returns for risk-free rate
  const adjustedReturns = validated.map((r) => r - rfPerPeriod);

  // Calculate downside deviation (non-annualized)
  const n = adjustedReturns.length;
  const negativeReturns = filter(adjustedReturns, (r) => r < 0);

  if (negativeReturns.length === 0) {
    const meanRet = mean(adjustedReturns);
    return meanRet > 0 ? Infinity : 0.0;
  }

  const squaredSum = sum(negativeReturns.map((r) => r * r));
  const downside = Math.sqrt(squaredSum / n);

  if (downside === 0) {
    const meanRet = mean(adjustedReturns);
    return meanRet > 0 ? Infinity : 0.0;
  }

  const meanReturn = mean(adjustedReturns);

  // Annualize
  return (meanReturn / downside) * Math.sqrt(annFactor);
}

/**
 * Calculate the Calmar ratio.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.periodsPerYear - Periods per year for CAGR calculation.
 * @returns Calmar ratio (CAGR / |max drawdown|).
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, -0.05, 0.03, 0.02];
 * calmar(returns, { periodsPerYear: 252 }); // ~5.0
 * ```
 */
export function calmar(
  returns: number[],
  options: { periodsPerYear?: number } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const config = getConfig();
  const ppy = options.periodsPerYear ?? config.periodsPerYear;

  const annualReturn = cagr(validated, { periodsPerYear: ppy });
  const mdd = maxDrawdown(validated);

  // maxDrawdown returns negative, take absolute value
  if (mdd === 0) {
    return annualReturn > 0 ? Infinity : 0.0;
  }

  return safeDivide(annualReturn, Math.abs(mdd), 0.0);
}

/**
 * Calculate the Omega ratio.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.threshold - Required return (annualized). Default: 0.
 * @param options.riskFreeRate - Risk-free rate (annualized). Default: 0.
 * @param options.periodsPerYear - Periods per year for threshold conversion.
 * @returns Omega ratio (probability-weighted gains / losses).
 *
 * @example
 * ```ts
 * const returns = [0.02, 0.01, -0.01, 0.03, -0.02];
 * omega(returns, { threshold: 0.0 }); // 2.0
 * ```
 */
export function omega(
  returns: number[],
  options: {
    threshold?: number;
    riskFreeRate?: number;
    periodsPerYear?: number;
  } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length < 2) {
    return NaN;
  }

  const { threshold = 0.0, riskFreeRate = 0.0, periodsPerYear } = options;

  if (threshold <= -1) {
    return NaN;
  }

  const config = getConfig();
  const annFactor = getAnnualizationFactor(
    periodsPerYear ?? config.periodsPerYear
  );

  // Adjust returns for risk-free rate
  const rfPerPeriod = riskFreeRate / annFactor;
  const adjustedReturns = validated.map((r) => r - rfPerPeriod);

  // Convert annualized threshold to per-period (QuantStats formula)
  let thresholdPerPeriod: number;
  if (annFactor === 1) {
    thresholdPerPeriod = threshold;
  } else {
    thresholdPerPeriod = Math.pow(1 + threshold, 1.0 / annFactor) - 1;
  }

  // Calculate gains and losses relative to threshold
  const returnsLessThresh = adjustedReturns.map((r) => r - thresholdPerPeriod);

  const gains = filter(returnsLessThresh, (r) => r > 0);
  const losses = filter(returnsLessThresh, (r) => r < 0);

  const sumGains = gains.length > 0 ? sum(gains) : 0.0;
  const sumLosses = losses.length > 0 ? Math.abs(sum(losses)) : 0.0;

  if (sumLosses === 0) {
    return sumGains > 0 ? Infinity : NaN;
  }

  return safeDivide(sumGains, sumLosses, NaN);
}

/**
 * Calculate gain-to-pain ratio.
 *
 * @param returns - Period returns.
 * @returns Sum of all returns / Sum of absolute losses.
 *
 * @example
 * ```ts
 * const returns = [0.02, 0.01, -0.01, 0.03, -0.02];
 * gainToPainRatio(returns); // 1.0
 * ```
 */
export function gainToPainRatio(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const totalReturn = sum(validated);
  const losses = filter(validated, (r) => r < 0);
  const sumAbsLosses = losses.length > 0 ? Math.abs(sum(losses)) : 0.0;

  if (sumAbsLosses === 0) {
    return totalReturn > 0 ? Infinity : 0.0;
  }

  return safeDivide(totalReturn, sumAbsLosses, 0.0);
}

/**
 * Calculate Ulcer Performance Index (UPI).
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.riskFreeRate - Risk-free rate (used as flat adjustment).
 * @returns UPI = (comp(returns) - rf) / Ulcer Index.
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, -0.01, 0.015, 0.005];
 * ulcerPerformanceIndex(returns, { riskFreeRate: 0.0 }); // ~5.0
 * ```
 */
export function ulcerPerformanceIndex(
  returns: number[],
  options: { riskFreeRate?: number } = {}
): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const config = getConfig();
  const rf = options.riskFreeRate ?? config.riskFreeRate;

  const totalReturn = comp(validated);
  const ui = ulcerIndex(validated);

  const excess = totalReturn - rf;

  if (ui === 0) {
    return excess > 0 ? Infinity : 0.0;
  }

  return safeDivide(excess, ui, 0.0);
}

/**
 * Calculate Kelly criterion (optimal bet sizing).
 *
 * @param returns - Period returns.
 * @returns Optimal fraction of capital to risk per trade.
 *
 * @example
 * ```ts
 * const returns = [0.02, -0.01, 0.03, -0.01, 0.02];
 * kellyCriterion(returns); // ~0.40
 * ```
 */
export function kellyCriterion(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  // Win rate
  const wins = filter(validated, (r) => r > 0);
  const losses = filter(validated, (r) => r < 0);

  const nTrades = validated.length;
  const nWins = wins.length;

  if (nTrades === 0) {
    return 0.0;
  }

  const w = nWins / nTrades;

  // Calculate payoff ratio
  const avgWin = wins.length > 0 ? mean(wins) : 0.0;
  const avgLoss = losses.length > 0 ? mean(losses) : 0.0;

  if (avgLoss === 0) {
    return w > 0 ? w : 0.0;
  }

  // Payoff ratio (avgLoss is negative, so take abs)
  const r = avgWin / Math.abs(avgLoss);

  // Handle edge case where payoff ratio is 0 (no wins)
  if (r === 0) {
    return -Infinity; // No wins means kelly suggests full loss
  }

  // Kelly formula
  return w - (1 - w) / r;
}

/**
 * Calculate tail ratio (right tail / left tail).
 *
 * @param returns - Period returns.
 * @returns Ratio of 95th percentile to 5th percentile (absolute).
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, -0.01, 0.05, -0.02];
 * tailRatio(returns); // ~2.5
 * ```
 */
export function tailRatio(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const rightTail = quantile(validated, 0.95);
  const leftTail = quantile(validated, 0.05);

  if (leftTail === 0) {
    return 0.0;
  }

  return safeDivide(Math.abs(rightTail), Math.abs(leftTail), 0.0);
}

/**
 * Calculate common sense ratio.
 *
 * @param returns - Period returns.
 * @returns Profit factor * Tail ratio.
 *
 * @example
 * ```ts
 * const returns = [0.02, -0.01, 0.03, -0.01, 0.02];
 * commonSenseRatio(returns); // ~6.25
 * ```
 */
export function commonSenseRatio(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  return profitFactor(validated) * tailRatio(validated);
}

/**
 * Calculate risk-return ratio.
 *
 * @param returns - Period returns.
 * @returns Mean return divided by standard deviation (not annualized).
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, -0.01, 0.015, 0.005];
 * riskReturnRatio(returns); // ~0.65
 * ```
 */
export function riskReturnRatio(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const meanRet = mean(validated);
  const stdRet = std(validated, 1);

  if (stdRet === 0) {
    return 0.0;
  }

  return meanRet / stdRet;
}

/**
 * Calculate recovery factor.
 *
 * @param returns - Period returns.
 * @returns Sum of returns divided by absolute max drawdown.
 *
 * @example
 * ```ts
 * const returns = [0.10, -0.05, 0.08, 0.05];
 * recoveryFactor(returns); // ~3.6
 * ```
 */
export function recoveryFactor(returns: number[]): number {
  const validated = validateReturns(returns, { allowEmpty: true });
  if (validated.length === 0) {
    return 0.0;
  }

  const total = sum(validated);
  const mdd = maxDrawdown(validated);

  if (mdd === 0) {
    return total > 0 ? Infinity : 0.0;
  }

  return safeDivide(Math.abs(total), Math.abs(mdd), 0.0);
}

// ===== BENCHMARK COMPARISON METRICS =====

/**
 * Calculate alpha and beta relative to a benchmark.
 *
 * @param returns - Strategy period returns.
 * @param benchmark - Benchmark period returns (same frequency).
 * @param options - Optional parameters.
 * @param options.periodsPerYear - Periods per year for annualization.
 * @returns { alpha, beta } where alpha is annualized excess return.
 *
 * @example
 * ```ts
 * const returns = [0.02, -0.01, 0.03, 0.01, -0.02];
 * const benchmark = [0.01, -0.02, 0.02, 0.01, -0.01];
 * const { alpha, beta } = greeks(returns, benchmark);
 * ```
 */
export function greeks(
  returns: number[],
  benchmark: number[],
  options: { periodsPerYear?: number } = {}
): Greeks {
  const validatedReturns = validateReturns(returns, { allowEmpty: true });
  const validatedBenchmark = validateReturns(benchmark, { allowEmpty: true });
  validateBenchmarkMatch(validatedReturns, validatedBenchmark);

  if (validatedReturns.length === 0 || validatedReturns.length < 2) {
    return { alpha: 0.0, beta: 0.0 };
  }

  const config = getConfig();
  const annFactor = getAnnualizationFactor(
    options.periodsPerYear ?? config.periodsPerYear
  );

  // Calculate beta (covariance / variance)
  const meanRet = mean(validatedReturns);
  const meanBench = mean(validatedBenchmark);

  // Covariance (population mean)
  const cov = mean(
    validatedReturns.map((r, i) => (r - meanRet) * (validatedBenchmark[i] - meanBench))
  );
  // Variance of benchmark (population)
  const varBench = mean(
    validatedBenchmark.map((b) => Math.pow(b - meanBench, 2))
  );

  let beta: number;
  if (varBench === 0) {
    beta = 0.0;
  } else {
    beta = cov / varBench;
  }

  // Alpha (annualized): alpha_period = mean_ret - beta * mean_bench
  const alphaPeriod = meanRet - beta * meanBench;
  const alpha = alphaPeriod * annFactor;

  return { alpha, beta };
}

/**
 * Calculate information ratio.
 *
 * @param returns - Strategy period returns.
 * @param benchmark - Benchmark period returns.
 * @returns Information ratio (active return / tracking error).
 *
 * @example
 * ```ts
 * const returns = [0.02, -0.01, 0.03, 0.01, -0.02];
 * const benchmark = [0.01, -0.02, 0.02, 0.01, -0.01];
 * informationRatio(returns, benchmark); // ~0.25
 * ```
 */
export function informationRatio(
  returns: number[],
  benchmark: number[]
): number {
  const validatedReturns = validateReturns(returns, { allowEmpty: true });
  const validatedBenchmark = validateReturns(benchmark, { allowEmpty: true });
  validateBenchmarkMatch(validatedReturns, validatedBenchmark);

  if (validatedReturns.length === 0 || validatedReturns.length < 2) {
    return 0.0;
  }

  // Active returns (tracking difference)
  const active = validatedReturns.map((r, i) => r - validatedBenchmark[i]);

  const meanActive = mean(active);
  const stdActive = std(active, 1);

  if (stdActive === 0) {
    return 0.0;
  }

  return meanActive / stdActive;
}

/**
 * Calculate R-squared (coefficient of determination).
 *
 * @param returns - Strategy period returns.
 * @param benchmark - Benchmark period returns.
 * @returns R-squared value (0 to 1).
 *
 * @example
 * ```ts
 * const returns = [0.02, -0.01, 0.03, 0.01, -0.02];
 * const benchmark = [0.01, -0.02, 0.02, 0.01, -0.01];
 * rSquared(returns, benchmark); // ~0.60
 * ```
 */
export function rSquared(returns: number[], benchmark: number[]): number {
  const validatedReturns = validateReturns(returns, { allowEmpty: true });
  const validatedBenchmark = validateReturns(benchmark, { allowEmpty: true });
  validateBenchmarkMatch(validatedReturns, validatedBenchmark);

  if (validatedReturns.length === 0 || validatedReturns.length < 2) {
    return 0.0;
  }

  const n = validatedReturns.length;

  // Calculate correlation using consistent sample formulas (ddof=1)
  const meanRet = mean(validatedReturns);
  const meanBench = mean(validatedBenchmark);

  // Sample covariance (N-1 denominator)
  const covSum = sum(
    validatedReturns.map(
      (r, i) => (r - meanRet) * (validatedBenchmark[i] - meanBench)
    )
  );
  const cov = covSum / (n - 1);

  // Sample standard deviations (ddof=1, N-1 denominator)
  const stdRet = std(validatedReturns, 1);
  const stdBench = std(validatedBenchmark, 1);

  if (stdRet === 0 || stdBench === 0) {
    return 0.0;
  }

  const correlation = cov / (stdRet * stdBench);
  return correlation * correlation;
}

/**
 * Calculate Treynor ratio.
 *
 * @param returns - Strategy period returns.
 * @param benchmark - Benchmark period returns.
 * @param options - Optional parameters.
 * @param options.riskFreeRate - Annualized risk-free rate.
 * @param options.periodsPerYear - Periods per year for annualization.
 * @returns Treynor ratio (excess return / beta).
 *
 * @example
 * ```ts
 * const returns = [0.02, -0.01, 0.03, 0.01, -0.02];
 * const benchmark = [0.01, -0.02, 0.02, 0.01, -0.01];
 * treynorRatio(returns, benchmark); // ~0.50
 * ```
 */
export function treynorRatio(
  returns: number[],
  benchmark: number[],
  options: { riskFreeRate?: number; periodsPerYear?: number } = {}
): number {
  const validatedReturns = validateReturns(returns, { allowEmpty: true });
  const validatedBenchmark = validateReturns(benchmark, { allowEmpty: true });
  validateBenchmarkMatch(validatedReturns, validatedBenchmark);

  if (validatedReturns.length === 0 || validatedReturns.length < 2) {
    return 0.0;
  }

  const config = getConfig();
  const rf = options.riskFreeRate ?? config.riskFreeRate;
  const ppy = options.periodsPerYear ?? config.periodsPerYear;

  // Calculate beta
  const { beta } = greeks(validatedReturns, validatedBenchmark, {
    periodsPerYear: ppy,
  });

  if (beta === 0) {
    const annualReturn = cagr(validatedReturns, { periodsPerYear: ppy });
    return annualReturn > rf ? Infinity : 0.0;
  }

  const annualReturn = cagr(validatedReturns, { periodsPerYear: ppy });
  const excessReturn = annualReturn - rf;

  return safeDivide(excessReturn, beta, 0.0);
}

/**
 * Calculate correlation with benchmark.
 *
 * @param returns - Strategy period returns.
 * @param benchmark - Benchmark period returns.
 * @returns Pearson correlation coefficient (-1 to 1).
 *
 * @example
 * ```ts
 * const returns = [0.02, -0.01, 0.03, 0.01, -0.02];
 * const benchmark = [0.01, -0.02, 0.02, 0.01, -0.01];
 * benchmarkCorrelation(returns, benchmark); // ~0.77
 * ```
 */
export function benchmarkCorrelation(
  returns: number[],
  benchmark: number[]
): number {
  const validatedReturns = validateReturns(returns, { allowEmpty: true });
  const validatedBenchmark = validateReturns(benchmark, { allowEmpty: true });
  validateBenchmarkMatch(validatedReturns, validatedBenchmark);

  if (validatedReturns.length === 0 || validatedReturns.length < 2) {
    return 0.0;
  }

  const n = validatedReturns.length;

  // Calculate correlation using consistent sample formulas (ddof=1)
  const meanRet = mean(validatedReturns);
  const meanBench = mean(validatedBenchmark);

  // Sample covariance (N-1 denominator)
  const covSum = sum(
    validatedReturns.map(
      (r, i) => (r - meanRet) * (validatedBenchmark[i] - meanBench)
    )
  );
  const cov = covSum / (n - 1);

  // Sample standard deviations (ddof=1, N-1 denominator)
  const stdRet = std(validatedReturns, 1);
  const stdBench = std(validatedBenchmark, 1);

  if (stdRet === 0 || stdBench === 0) {
    return 0.0;
  }

  return cov / (stdRet * stdBench);
}
