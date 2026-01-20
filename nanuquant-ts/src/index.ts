/**
 * NanuQuant - Quantitative finance metrics library.
 *
 * A zero-dependency TypeScript port of the NanuQuant Python library.
 *
 * @example
 * ```ts
 * import { sharpe, maxDrawdown, cagr } from 'nanuquant';
 *
 * const returns = [0.01, -0.02, 0.015, -0.01, 0.02];
 * console.log('Sharpe:', sharpe(returns));
 * console.log('Max DD:', maxDrawdown(returns));
 * console.log('CAGR:', cagr(returns, { periodsPerYear: 252 }));
 * ```
 */

// Types
export type {
  FrequencyType,
  RollingWindowType,
  Greeks,
  RollingGreeks,
} from './types';

export {
  ANNUALIZATION_PERIODS,
  VAR_SIGMA_MAP,
  ROLLING_WINDOW_DAYS,
} from './types';

// Config
export { MetricsConfig, getConfig, setConfig, resetConfig } from './config';

// Exceptions
export {
  MetricsError,
  EmptySeriesError,
  InsufficientDataError,
  BenchmarkMismatchError,
} from './exceptions';

// Validation
export {
  validateReturns,
  validateMinLength,
  validateBenchmarkMatch,
} from './core/validation';

// Utilities
export {
  mean,
  std,
  sum,
  cumProd,
  cumMax,
  quantile,
  safeDivide,
  getAnnualizationFactor,
  filter,
  count,
  max,
  min,
} from './core/utils';

// Returns metrics
export {
  comp,
  cagr,
  avgReturn,
  avgWin,
  avgLoss,
  best,
  worst,
  winRate,
  payoffRatio,
  profitFactor,
  consecutiveWins,
  consecutiveLosses,
} from './core/returns';

// Risk metrics
export {
  volatility,
  varMetric,
  cvar,
  toDrawdownSeries,
  maxDrawdown,
  ulcerIndex,
  downsideDeviation,
} from './core/risk';

// Performance metrics
export {
  sharpe,
  sortino,
  calmar,
  omega,
  gainToPainRatio,
  ulcerPerformanceIndex,
  kellyCriterion,
  tailRatio,
  commonSenseRatio,
  riskReturnRatio,
  recoveryFactor,
  greeks,
  informationRatio,
  rSquared,
  treynorRatio,
  benchmarkCorrelation,
} from './core/performance';

// Correlation metrics
export {
  correlation,
  covariance,
  downsideCorrelation,
  upsideCorrelation,
  rollingCorrelation,
} from './core/correlation';

// Distribution metrics
export {
  skewness,
  kurtosis,
  expectedReturn,
  geometricMean,
  outlierWinRatio,
  outlierLossRatio,
  outliers,
  removeOutliers,
} from './core/distribution';

// Rolling metrics
export {
  rollingVolatility,
  rollingSharpe,
  rollingSortino,
  rollingBeta,
  rollingGreeks,
} from './core/rolling';

// Timeseries functions
export type {
  HistogramBin,
  PeriodStats,
  DistributionResult,
  MonthlyReturnsMatrix,
  PeriodReturn,
} from './core/timeseries';

export {
  cumulativeReturns,
  equityCurve,
  logReturns,
  simpleReturns,
  histogram,
  yearlyReturns,
  monthlyReturns,
  weeklyReturns,
  quarterlyReturns,
  distribution,
  periodReturns,
  monthlyReturnsMatrix,
} from './core/timeseries';

// Stats
export { normPdf, normCdf, normPpf, normPpfAccurate } from './stats/normal';
