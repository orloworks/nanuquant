/**
 * Input validation for nanuquant.
 *
 * This module provides validation utilities for returns data with explicit policies
 * for null handling and dtype enforcement.
 */

import {
  BenchmarkMismatchError,
  EmptySeriesError,
  InsufficientDataError,
} from '../exceptions';

/**
 * Validate a returns array.
 *
 * @param data - Array to validate.
 * @param options - Validation options.
 * @param options.allowEmpty - If true, empty arrays will not raise an error.
 * @param options.dropNaN - If true, NaN values are filtered from the array.
 * @returns The validated (and optionally cleaned) array.
 * @throws EmptySeriesError if data is empty and allowEmpty is false.
 */
export function validateReturns(
  data: number[],
  options: { allowEmpty?: boolean; dropNaN?: boolean } = {}
): number[] {
  const { allowEmpty = false, dropNaN = true } = options;

  // Drop NaN values if requested (default behavior)
  let result = dropNaN ? data.filter((x) => !Number.isNaN(x)) : data;

  if (result.length === 0 && !allowEmpty) {
    throw new EmptySeriesError('Returns series is empty');
  }

  // Verify all values are numbers
  for (const value of result) {
    if (typeof value !== 'number') {
      throw new TypeError(`Expected number, got ${typeof value}`);
    }
  }

  return result;
}

/**
 * Validate that data has minimum required length.
 *
 * @param data - Array to validate.
 * @param minLength - Minimum required length.
 * @param metric - Name of the metric for error message.
 * @throws InsufficientDataError if data length is less than minLength.
 */
export function validateMinLength(
  data: number[],
  minLength: number,
  metric: string = ''
): void {
  if (data.length < minLength) {
    throw new InsufficientDataError(minLength, data.length, metric);
  }
}

/**
 * Validate that strategy and benchmark have matching lengths.
 *
 * @param strategy - Strategy returns.
 * @param benchmark - Benchmark returns.
 * @throws BenchmarkMismatchError if lengths don't match.
 */
export function validateBenchmarkMatch(
  strategy: number[],
  benchmark: number[]
): void {
  if (strategy.length !== benchmark.length) {
    throw new BenchmarkMismatchError(strategy.length, benchmark.length);
  }
}

/**
 * Validate that value is positive.
 *
 * @param value - Value to check.
 * @param name - Name for error message.
 * @throws ValueError if value is not positive.
 */
export function validatePositive(value: number, name: string = 'value'): void {
  if (value <= 0) {
    throw new Error(`${name} must be positive, got ${value}`);
  }
}

/**
 * Validate that value is a valid probability (0-1).
 *
 * @param value - Value to check.
 * @param name - Name for error message.
 * @throws ValueError if value is not in [0, 1].
 */
export function validateProbability(
  value: number,
  name: string = 'probability'
): void {
  if (value < 0 || value > 1) {
    throw new Error(`${name} must be between 0 and 1, got ${value}`);
  }
}
