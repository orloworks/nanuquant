/**
 * Test helpers for nanuquant tests.
 */

import { readFileSync } from 'fs';
import { join } from 'path';

/**
 * Tolerance levels for comparing numerical results.
 */
export const TOLERANCES = {
  EXACT: { rtol: 1e-10, atol: 0 },
  TIGHT: { rtol: 1e-8, atol: 1e-12 },
  LOOSE: { rtol: 1e-6, atol: 1e-10 },
  STAT: { rtol: 1e-4, atol: 1e-8 },
  // Very loose tolerance for statistical moments (skewness/kurtosis)
  // Different libraries use different bias correction formulas
  STATISTICAL: { rtol: 0.1, atol: 0.01 },
};

/**
 * Load JSON fixture from file.
 */
export function loadFixture<T>(filename: string): T {
  const fixturePath = join(__dirname, 'fixtures', filename);
  const content = readFileSync(fixturePath, 'utf-8');
  return JSON.parse(content) as T;
}

/**
 * Load expected results from fixture.
 */
export function loadExpected<T>(module: string): T {
  return loadFixture<T>(`expected/${module}.json`);
}

/**
 * Load input test data.
 */
export interface TestInputs {
  returns: number[];
  benchmark: number[];
  dates: string[];
  seed: number;
  n: number;
}

export function loadInputs(): TestInputs {
  return loadFixture<TestInputs>('inputs.json');
}

/**
 * Check if two numbers are approximately equal.
 */
export function isClose(
  actual: number,
  expected: number,
  tolerance: { rtol: number; atol: number } = TOLERANCES.TIGHT
): boolean {
  if (!Number.isFinite(actual) && !Number.isFinite(expected)) {
    // Both are Infinity or -Infinity
    return actual === expected;
  }
  if (Number.isNaN(actual) && Number.isNaN(expected)) {
    return true;
  }
  if (Number.isNaN(actual) || Number.isNaN(expected)) {
    return false;
  }
  if (!Number.isFinite(actual) || !Number.isFinite(expected)) {
    return false;
  }

  const diff = Math.abs(actual - expected);
  const relTol = tolerance.rtol * Math.abs(expected);
  return diff <= Math.max(tolerance.atol, relTol);
}

/**
 * Assert two numbers are approximately equal.
 */
export function assertClose(
  actual: number,
  expected: number,
  tolerance: { rtol: number; atol: number } = TOLERANCES.TIGHT,
  message?: string
): void {
  if (!isClose(actual, expected, tolerance)) {
    const diff = Math.abs(actual - expected);
    const relErr = expected !== 0 ? diff / Math.abs(expected) : diff;
    throw new Error(
      `${message || 'Values not close'}: actual=${actual}, expected=${expected}, diff=${diff}, relErr=${relErr}`
    );
  }
}

/**
 * Assert two arrays are approximately equal element-wise.
 */
export function assertArrayClose(
  actual: number[],
  expected: number[],
  tolerance: { rtol: number; atol: number } = TOLERANCES.TIGHT,
  message?: string
): void {
  if (actual.length !== expected.length) {
    throw new Error(
      `${message || 'Array lengths differ'}: actual.length=${actual.length}, expected.length=${expected.length}`
    );
  }

  for (let i = 0; i < actual.length; i++) {
    if (!isClose(actual[i], expected[i], tolerance)) {
      const diff = Math.abs(actual[i] - expected[i]);
      const relErr = expected[i] !== 0 ? diff / Math.abs(expected[i]) : diff;
      throw new Error(
        `${message || 'Array values not close'} at index ${i}: actual=${actual[i]}, expected=${expected[i]}, diff=${diff}, relErr=${relErr}`
      );
    }
  }
}

/**
 * Handle null values in expected (representing NaN in Python).
 */
export function normalizeExpected(value: number | null): number {
  return value === null ? NaN : value;
}

/**
 * Normalize an array of expected values.
 */
export function normalizeExpectedArray(values: (number | null)[]): number[] {
  return values.map(normalizeExpected);
}
