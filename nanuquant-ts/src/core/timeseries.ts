/**
 * Timeseries analysis for nanuquant.
 *
 * This module provides functions that return arrays rather than scalars,
 * designed for visualization and multi-period analysis.
 */

import { compoundReturns as utilCompoundReturns, sum, mean, std, quantile } from './utils';
import { validateReturns } from './validation';

/**
 * Histogram result type.
 */
export interface HistogramBin {
  binStart: number;
  binEnd: number;
  binCenter: number;
  count: number;
  frequency: number;
  density?: number;
}

/**
 * Period statistics type.
 */
export interface PeriodStats {
  values: number[];
  outliers: number[];
  mean: number;
  std: number;
  min: number;
  max: number;
  count: number;
}

/**
 * Distribution result type.
 */
export interface DistributionResult {
  Daily?: PeriodStats;
  Weekly?: PeriodStats;
  Monthly?: PeriodStats;
  Quarterly?: PeriodStats;
  Yearly?: PeriodStats;
}

/**
 * Monthly returns matrix type.
 */
export interface MonthlyReturnsMatrix {
  months: number[];
  years: number[];
  data: Map<string, number>; // "year-month" -> return
}

/**
 * Period return type for YTD/MTD/WTD/QTD.
 */
export interface PeriodReturn {
  period: string;
  return: number;
  startDate?: Date;
  endDate?: Date;
}

/**
 * Calculate cumulative compounded returns (equity curve).
 *
 * @param returns - Period returns.
 * @returns Array of cumulative returns (growth from $1).
 *
 * @example
 * ```ts
 * const returns = [0.01, 0.02, -0.01, 0.03];
 * cumulativeReturns(returns);
 * // [0.01, 0.0302, 0.019898, 0.050495]
 * ```
 */
export function cumulativeReturns(returns: number[]): number[] {
  const validated = validateReturns(returns, { allowEmpty: true });

  if (validated.length === 0) {
    return [];
  }

  // Use the utility function that properly calculates compound returns
  return utilCompoundReturns(validated);
}

/**
 * Alias for cumulativeReturns.
 *
 * @param returns - Period returns.
 * @returns Array of cumulative returns (equity curve).
 */
export function equityCurve(returns: number[]): number[] {
  return cumulativeReturns(returns);
}

/**
 * Convert simple returns to log returns.
 *
 * @param returns - Simple returns (e.g., 0.05 for 5%).
 * @returns Log returns: ln(1 + r).
 *
 * @example
 * ```ts
 * const simple = [0.05, -0.03, 0.02];
 * logReturns(simple);
 * // [0.0488, -0.0305, 0.0198]
 * ```
 */
export function logReturns(returns: number[]): number[] {
  const validated = validateReturns(returns, { allowEmpty: true });

  return validated.map((r) => {
    if (r <= -1) {
      return -Infinity;
    }
    return Math.log(1 + r);
  });
}

/**
 * Convert log returns to simple returns.
 *
 * @param logRets - Log returns.
 * @returns Simple returns: exp(r) - 1.
 *
 * @example
 * ```ts
 * const log = [0.0488, -0.0305, 0.0198];
 * simpleReturns(log);
 * // [0.05, -0.03, 0.02]
 * ```
 */
export function simpleReturns(logRets: number[]): number[] {
  const validated = validateReturns(logRets, { allowEmpty: true });

  return validated.map((r) => Math.exp(r) - 1);
}

/**
 * Calculate histogram of returns for distribution visualization.
 *
 * @param returns - Period returns.
 * @param options - Optional parameters.
 * @param options.bins - Number of bins. Default: 50.
 * @param options.density - If true, normalize to density (area sums to 1). Default: false.
 * @returns Array of histogram bins.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, -0.01, 0.02];
 * histogram(returns, { bins: 5 });
 * ```
 */
export function histogram(
  returns: number[],
  options: { bins?: number; density?: boolean } = {}
): HistogramBin[] {
  const validated = validateReturns(returns, { allowEmpty: true });
  const { bins = 50, density = false } = options;

  if (validated.length === 0) {
    return [];
  }

  const minVal = Math.min(...validated);
  const maxVal = Math.max(...validated);

  // Handle edge case where all values are the same
  if (minVal === maxVal) {
    const bin: HistogramBin = {
      binStart: minVal - 0.5,
      binEnd: maxVal + 0.5,
      binCenter: minVal,
      count: validated.length,
      frequency: 1.0,
    };
    if (density) {
      bin.density = 1.0;
    }
    return [bin];
  }

  const binWidth = (maxVal - minVal) / bins;
  const result: HistogramBin[] = [];
  const total = validated.length;

  // Initialize bins
  for (let i = 0; i < bins; i++) {
    const binStart = minVal + i * binWidth;
    const binEnd = binStart + binWidth;
    result.push({
      binStart,
      binEnd,
      binCenter: (binStart + binEnd) / 2,
      count: 0,
      frequency: 0,
    });
  }

  // Count values in each bin
  for (const value of validated) {
    let binIndex = Math.floor((value - minVal) / binWidth);
    // Handle edge case where value equals maxVal
    if (binIndex >= bins) {
      binIndex = bins - 1;
    }
    result[binIndex].count++;
  }

  // Calculate frequency and optionally density
  for (const bin of result) {
    bin.frequency = bin.count / total;
    if (density) {
      bin.density = bin.frequency / binWidth;
    }
  }

  return result;
}

/**
 * Identify outliers using IQR method.
 *
 * @param values - Array of values.
 * @returns Array of outlier values.
 */
function identifyOutliersIQR(values: number[]): number[] {
  if (values.length < 4) {
    return [];
  }

  const q1 = quantile(values, 0.25);
  const q3 = quantile(values, 0.75);
  const iqr = q3 - q1;
  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;

  return values.filter((v) => v < lowerBound || v > upperBound);
}

/**
 * Compute statistics for a series.
 *
 * @param values - Array of values.
 * @returns Period statistics.
 */
function computeStats(values: number[]): PeriodStats {
  if (values.length === 0) {
    return {
      values: [],
      outliers: [],
      mean: 0,
      std: 0,
      min: 0,
      max: 0,
      count: 0,
    };
  }

  const outliers = identifyOutliersIQR(values);

  return {
    values,
    outliers,
    mean: mean(values),
    std: values.length > 1 ? std(values, 1) : 0,
    min: Math.min(...values),
    max: Math.max(...values),
    count: values.length,
  };
}

/**
 * Group returns by a date key function and optionally compound.
 *
 * @param returns - Period returns.
 * @param dates - Dates corresponding to returns.
 * @param keyFn - Function to extract grouping key from date.
 * @param compounded - Whether to compound returns within group.
 * @returns Map of group key to aggregated return.
 */
function groupReturns(
  returns: number[],
  dates: Date[],
  keyFn: (d: Date) => string,
  compounded: boolean
): Map<string, number> {
  const groups = new Map<string, number[]>();

  for (let i = 0; i < returns.length; i++) {
    const key = keyFn(dates[i]);
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key)!.push(returns[i]);
  }

  const result = new Map<string, number>();
  for (const [key, values] of groups) {
    if (compounded) {
      // Compound: (1 + r1) * (1 + r2) * ... - 1
      const product = values.reduce((acc, r) => acc * (1 + r), 1);
      result.set(key, product - 1);
    } else {
      result.set(key, sum(values));
    }
  }

  return result;
}

/**
 * Calculate yearly returns.
 *
 * @param returns - Period returns.
 * @param dates - Dates corresponding to returns.
 * @param options - Optional parameters.
 * @param options.compounded - If true, compound returns within each year. Default: true.
 * @returns Map of year to yearly return.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, ...];
 * const dates = [new Date('2020-01-01'), new Date('2020-01-02'), ...];
 * yearlyReturns(returns, dates);
 * // Map { '2020' => 0.15, '2021' => 0.08, ... }
 * ```
 */
export function yearlyReturns(
  returns: number[],
  dates: Date[],
  options: { compounded?: boolean } = {}
): Map<string, number> {
  const validated = validateReturns(returns, { allowEmpty: true });
  const { compounded = true } = options;

  if (validated.length === 0 || dates.length === 0) {
    return new Map();
  }

  if (validated.length !== dates.length) {
    throw new Error('Returns and dates must have the same length');
  }

  return groupReturns(validated, dates, (d) => d.getFullYear().toString(), compounded);
}

/**
 * Calculate monthly returns.
 *
 * @param returns - Period returns.
 * @param dates - Dates corresponding to returns.
 * @param options - Optional parameters.
 * @param options.compounded - If true, compound returns within each month. Default: true.
 * @returns Map of "year-month" to monthly return.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, ...];
 * const dates = [new Date('2020-01-01'), new Date('2020-01-02'), ...];
 * monthlyReturns(returns, dates);
 * // Map { '2020-1' => 0.05, '2020-2' => 0.03, ... }
 * ```
 */
export function monthlyReturns(
  returns: number[],
  dates: Date[],
  options: { compounded?: boolean } = {}
): Map<string, number> {
  const validated = validateReturns(returns, { allowEmpty: true });
  const { compounded = true } = options;

  if (validated.length === 0 || dates.length === 0) {
    return new Map();
  }

  if (validated.length !== dates.length) {
    throw new Error('Returns and dates must have the same length');
  }

  return groupReturns(
    validated,
    dates,
    (d) => `${d.getFullYear()}-${d.getMonth() + 1}`,
    compounded
  );
}

/**
 * Get ISO week number from date.
 *
 * @param date - Date to get week number from.
 * @returns ISO week number (1-53).
 */
function getISOWeek(date: Date): number {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const dayNum = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  return Math.ceil(((d.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
}

/**
 * Calculate weekly returns.
 *
 * @param returns - Period returns.
 * @param dates - Dates corresponding to returns.
 * @param options - Optional parameters.
 * @param options.compounded - If true, compound returns within each week. Default: true.
 * @returns Map of "year-week" to weekly return.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, ...];
 * const dates = [new Date('2020-01-01'), new Date('2020-01-02'), ...];
 * weeklyReturns(returns, dates);
 * // Map { '2020-1' => 0.02, '2020-2' => 0.01, ... }
 * ```
 */
export function weeklyReturns(
  returns: number[],
  dates: Date[],
  options: { compounded?: boolean } = {}
): Map<string, number> {
  const validated = validateReturns(returns, { allowEmpty: true });
  const { compounded = true } = options;

  if (validated.length === 0 || dates.length === 0) {
    return new Map();
  }

  if (validated.length !== dates.length) {
    throw new Error('Returns and dates must have the same length');
  }

  return groupReturns(
    validated,
    dates,
    (d) => `${d.getFullYear()}-${getISOWeek(d)}`,
    compounded
  );
}

/**
 * Calculate quarterly returns.
 *
 * @param returns - Period returns.
 * @param dates - Dates corresponding to returns.
 * @param options - Optional parameters.
 * @param options.compounded - If true, compound returns within each quarter. Default: true.
 * @returns Map of "year-Q#" to quarterly return.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, ...];
 * const dates = [new Date('2020-01-01'), new Date('2020-01-02'), ...];
 * quarterlyReturns(returns, dates);
 * // Map { '2020-Q1' => 0.08, '2020-Q2' => 0.05, ... }
 * ```
 */
export function quarterlyReturns(
  returns: number[],
  dates: Date[],
  options: { compounded?: boolean } = {}
): Map<string, number> {
  const validated = validateReturns(returns, { allowEmpty: true });
  const { compounded = true } = options;

  if (validated.length === 0 || dates.length === 0) {
    return new Map();
  }

  if (validated.length !== dates.length) {
    throw new Error('Returns and dates must have the same length');
  }

  return groupReturns(
    validated,
    dates,
    (d) => `${d.getFullYear()}-Q${Math.floor(d.getMonth() / 3) + 1}`,
    compounded
  );
}

/**
 * Analyze return distribution across multiple time periods.
 *
 * @param returns - Period returns.
 * @param dates - Dates corresponding to returns.
 * @param options - Optional parameters.
 * @param options.compounded - If true, compound returns within each period. Default: true.
 * @returns Distribution statistics for Daily, Weekly, Monthly, Quarterly, Yearly.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, ...];
 * const dates = [new Date('2020-01-01'), new Date('2020-01-02'), ...];
 * distribution(returns, dates);
 * // { Daily: {...}, Weekly: {...}, Monthly: {...}, ... }
 * ```
 */
export function distribution(
  returns: number[],
  dates: Date[],
  options: { compounded?: boolean } = {}
): DistributionResult {
  const validated = validateReturns(returns, { allowEmpty: true });
  const { compounded = true } = options;

  if (validated.length === 0) {
    return {};
  }

  if (validated.length !== dates.length) {
    throw new Error('Returns and dates must have the same length');
  }

  const result: DistributionResult = {};

  // Daily - just the raw returns
  result.Daily = computeStats(validated);

  // Weekly
  const weeklyMap = weeklyReturns(validated, dates, { compounded });
  result.Weekly = computeStats(Array.from(weeklyMap.values()));

  // Monthly
  const monthlyMap = monthlyReturns(validated, dates, { compounded });
  result.Monthly = computeStats(Array.from(monthlyMap.values()));

  // Quarterly
  const quarterlyMap = quarterlyReturns(validated, dates, { compounded });
  result.Quarterly = computeStats(Array.from(quarterlyMap.values()));

  // Yearly
  const yearlyMap = yearlyReturns(validated, dates, { compounded });
  result.Yearly = computeStats(Array.from(yearlyMap.values()));

  return result;
}

/**
 * Calculate period-to-date returns (YTD, MTD, WTD, QTD).
 *
 * @param returns - Period returns.
 * @param dates - Dates corresponding to returns.
 * @param options - Optional parameters.
 * @param options.referenceDate - Reference date for calculating period returns. Default: last date.
 * @param options.compounded - If true, compound returns within period. Default: true.
 * @returns Object with YTD, MTD, WTD, QTD returns.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, ...];
 * const dates = [new Date('2020-01-01'), new Date('2020-01-02'), ...];
 * periodReturns(returns, dates);
 * // { YTD: 0.08, MTD: 0.02, WTD: 0.01, QTD: 0.05 }
 * ```
 */
export function periodReturns(
  returns: number[],
  dates: Date[],
  options: { referenceDate?: Date; compounded?: boolean } = {}
): { YTD: number; MTD: number; WTD: number; QTD: number } {
  const validated = validateReturns(returns, { allowEmpty: true });
  const { compounded = true } = options;

  if (validated.length === 0 || dates.length === 0) {
    return { YTD: 0, MTD: 0, WTD: 0, QTD: 0 };
  }

  if (validated.length !== dates.length) {
    throw new Error('Returns and dates must have the same length');
  }

  const referenceDate = options.referenceDate || dates[dates.length - 1];
  const refYear = referenceDate.getFullYear();
  const refMonth = referenceDate.getMonth();
  const refQuarter = Math.floor(refMonth / 3);
  const refWeek = getISOWeek(referenceDate);

  // Filter returns for each period
  const ytdReturns: number[] = [];
  const mtdReturns: number[] = [];
  const wtdReturns: number[] = [];
  const qtdReturns: number[] = [];

  for (let i = 0; i < validated.length; i++) {
    const d = dates[i];
    if (d > referenceDate) continue;

    // YTD: same year
    if (d.getFullYear() === refYear) {
      ytdReturns.push(validated[i]);
    }

    // MTD: same year and month
    if (d.getFullYear() === refYear && d.getMonth() === refMonth) {
      mtdReturns.push(validated[i]);
    }

    // WTD: same ISO week
    if (d.getFullYear() === refYear && getISOWeek(d) === refWeek) {
      wtdReturns.push(validated[i]);
    }

    // QTD: same year and quarter
    if (d.getFullYear() === refYear && Math.floor(d.getMonth() / 3) === refQuarter) {
      qtdReturns.push(validated[i]);
    }
  }

  const aggregate = (rets: number[]) => {
    if (rets.length === 0) return 0;
    if (compounded) {
      return rets.reduce((acc, r) => acc * (1 + r), 1) - 1;
    }
    return sum(rets);
  };

  return {
    YTD: aggregate(ytdReturns),
    MTD: aggregate(mtdReturns),
    WTD: aggregate(wtdReturns),
    QTD: aggregate(qtdReturns),
  };
}

/**
 * Create a monthly returns matrix for tearsheet display.
 *
 * @param returns - Period returns.
 * @param dates - Dates corresponding to returns.
 * @param options - Optional parameters.
 * @param options.compounded - If true, compound returns within each month. Default: true.
 * @returns Matrix with months as rows, years as columns.
 *
 * @example
 * ```ts
 * const returns = [0.01, -0.02, 0.015, ...];
 * const dates = [new Date('2020-01-01'), new Date('2020-01-02'), ...];
 * const matrix = monthlyReturnsMatrix(returns, dates);
 * // Access: matrix.data.get('2020-1') for January 2020 return
 * ```
 */
export function monthlyReturnsMatrix(
  returns: number[],
  dates: Date[],
  options: { compounded?: boolean } = {}
): MonthlyReturnsMatrix {
  const validated = validateReturns(returns, { allowEmpty: true });
  const { compounded = true } = options;

  if (validated.length === 0 || dates.length === 0) {
    return { months: [], years: [], data: new Map() };
  }

  if (validated.length !== dates.length) {
    throw new Error('Returns and dates must have the same length');
  }

  const data = monthlyReturns(validated, dates, { compounded });

  // Extract unique months and years
  const monthsSet = new Set<number>();
  const yearsSet = new Set<number>();

  for (let i = 0; i < dates.length; i++) {
    monthsSet.add(dates[i].getMonth() + 1);
    yearsSet.add(dates[i].getFullYear());
  }

  const months = Array.from(monthsSet).sort((a, b) => a - b);
  const years = Array.from(yearsSet).sort((a, b) => a - b);

  return { months, years, data };
}
