/**
 * Utility functions for nanuquant.
 */

import { ANNUALIZATION_PERIODS, FrequencyType } from '../types';

/**
 * Get annualization factor for return calculations.
 *
 * @param periodsPerYear - Explicit periods per year. Takes precedence over frequency.
 * @param frequency - Data frequency to use for lookup. Default is "D" (daily).
 * @returns Annualization factor.
 */
export function getAnnualizationFactor(
  periodsPerYear?: number,
  frequency?: FrequencyType
): number {
  if (periodsPerYear !== undefined) {
    return periodsPerYear;
  }
  const freq = frequency ?? 'D';
  return ANNUALIZATION_PERIODS[freq];
}

/**
 * Safely divide, returning default for zero denominator.
 *
 * @param numerator - Numerator value.
 * @param denominator - Denominator value.
 * @param defaultValue - Value to return when denominator is zero.
 * @returns Division result or default.
 */
export function safeDivide(
  numerator: number,
  denominator: number,
  defaultValue: number = 0.0
): number {
  if (denominator === 0) {
    return defaultValue;
  }
  return numerator / denominator;
}

/**
 * Calculate the arithmetic mean of an array.
 *
 * @param arr - Input array.
 * @returns Mean value, or 0 if empty.
 */
export function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((sum, x) => sum + x, 0) / arr.length;
}

/**
 * Calculate the sum of an array.
 *
 * @param arr - Input array.
 * @returns Sum of all elements.
 */
export function sum(arr: number[]): number {
  return arr.reduce((acc, x) => acc + x, 0);
}

/**
 * Calculate sample or population standard deviation.
 *
 * @param arr - Input array.
 * @param ddof - Delta degrees of freedom. Default 1 (sample std).
 * @returns Standard deviation, or 0 if insufficient data.
 */
export function std(arr: number[], ddof: number = 1): number {
  if (arr.length <= ddof) return 0;
  const m = mean(arr);
  const squaredDiffs = arr.map((x) => (x - m) ** 2);
  const variance = sum(squaredDiffs) / (arr.length - ddof);
  return Math.sqrt(variance);
}

/**
 * Calculate variance.
 *
 * @param arr - Input array.
 * @param ddof - Delta degrees of freedom. Default 1 (sample variance).
 * @returns Variance, or 0 if insufficient data.
 */
export function variance(arr: number[], ddof: number = 1): number {
  if (arr.length <= ddof) return 0;
  const m = mean(arr);
  const squaredDiffs = arr.map((x) => (x - m) ** 2);
  return sum(squaredDiffs) / (arr.length - ddof);
}

/**
 * Calculate quantile using linear interpolation.
 *
 * @param arr - Input array (will be sorted).
 * @param q - Quantile value between 0 and 1.
 * @returns Quantile value.
 */
export function quantile(arr: number[], q: number): number {
  if (arr.length === 0) return NaN;
  if (arr.length === 1) return arr[0];

  const sorted = [...arr].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const lower = Math.floor(pos);
  const upper = Math.ceil(pos);

  if (lower === upper) {
    return sorted[lower];
  }

  // Linear interpolation
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (pos - lower);
}

/**
 * Get minimum value of array.
 *
 * @param arr - Input array.
 * @returns Minimum value, or NaN if empty.
 */
export function min(arr: number[]): number {
  if (arr.length === 0) return NaN;
  return Math.min(...arr);
}

/**
 * Get maximum value of array.
 *
 * @param arr - Input array.
 * @returns Maximum value, or NaN if empty.
 */
export function max(arr: number[]): number {
  if (arr.length === 0) return NaN;
  return Math.max(...arr);
}

/**
 * Calculate cumulative product of (1 + returns).
 *
 * @param arr - Input array of returns.
 * @returns Array of cumulative products.
 */
export function cumProd(arr: number[]): number[] {
  const result: number[] = [];
  let prod = 1;
  for (const val of arr) {
    prod *= 1 + val;
    result.push(prod);
  }
  return result;
}

/**
 * Calculate cumulative maximum.
 *
 * @param arr - Input array.
 * @returns Array of running maximums.
 */
export function cumMax(arr: number[]): number[] {
  const result: number[] = [];
  let maxVal = -Infinity;
  for (const val of arr) {
    maxVal = Math.max(maxVal, val);
    result.push(maxVal);
  }
  return result;
}

/**
 * Calculate cumulative sum.
 *
 * @param arr - Input array.
 * @returns Array of running sums.
 */
export function cumSum(arr: number[]): number[] {
  const result: number[] = [];
  let sumVal = 0;
  for (const val of arr) {
    sumVal += val;
    result.push(sumVal);
  }
  return result;
}

/**
 * Calculate compound returns series (cumulative compounded returns).
 *
 * @param returns - Period returns (not prices).
 * @returns Cumulative compounded returns.
 */
export function compoundReturns(returns: number[]): number[] {
  return cumProd(returns).map((x) => x - 1);
}

/**
 * Convert simple returns to log returns.
 *
 * @param returns - Simple (arithmetic) returns.
 * @returns Log (geometric) returns.
 */
export function logReturns(returns: number[]): number[] {
  return returns.map((r) => Math.log(1 + r));
}

/**
 * Convert log returns to simple returns.
 *
 * @param logRets - Log returns.
 * @returns Simple (arithmetic) returns.
 */
export function simpleReturns(logRets: number[]): number[] {
  return logRets.map((r) => Math.exp(r) - 1);
}

/**
 * Filter array elements that satisfy a predicate.
 *
 * @param arr - Input array.
 * @param predicate - Function to test each element.
 * @returns Filtered array.
 */
export function filter(
  arr: number[],
  predicate: (x: number) => boolean
): number[] {
  return arr.filter(predicate);
}

/**
 * Count array elements that satisfy a predicate.
 *
 * @param arr - Input array.
 * @param predicate - Function to test each element.
 * @returns Count of elements satisfying predicate.
 */
export function count(
  arr: number[],
  predicate: (x: number) => boolean
): number {
  return arr.filter(predicate).length;
}
