/**
 * Tests for correlation metrics.
 */

import { describe, it, expect } from 'vitest';
import {
  correlation,
  covariance,
  downsideCorrelation,
  upsideCorrelation,
  rollingCorrelation,
} from '../src/core/correlation';
import { loadInputs, loadExpected, assertClose, TOLERANCES } from './helpers';

interface CorrelationExpected {
  correlation: number;
  covariance: number;
}

describe('Correlation Metrics', () => {
  const inputs = loadInputs();
  const expected = loadExpected<CorrelationExpected>('correlation');
  const returns = inputs.returns;
  const benchmark = inputs.benchmark;

  it('correlation matches Python/NumPy', () => {
    const result = correlation(returns, benchmark);
    assertClose(result, expected.correlation, TOLERANCES.TIGHT, 'correlation');
  });

  it('covariance matches Python/NumPy', () => {
    const result = covariance(returns, benchmark);
    assertClose(result, expected.covariance, TOLERANCES.TIGHT, 'covariance');
  });

  it('downsideCorrelation returns number', () => {
    const result = downsideCorrelation(returns, benchmark);
    expect(typeof result).toBe('number');
    // Downside correlation should be between -1 and 1 (or NaN)
    if (!Number.isNaN(result)) {
      expect(result).toBeGreaterThanOrEqual(-1);
      expect(result).toBeLessThanOrEqual(1);
    }
  });

  it('upsideCorrelation returns number', () => {
    const result = upsideCorrelation(returns, benchmark);
    expect(typeof result).toBe('number');
    // Upside correlation should be between -1 and 1 (or NaN)
    if (!Number.isNaN(result)) {
      expect(result).toBeGreaterThanOrEqual(-1);
      expect(result).toBeLessThanOrEqual(1);
    }
  });

  it('rollingCorrelation returns correct length', () => {
    const windowSize = 20;
    const result = rollingCorrelation(returns, benchmark, windowSize);
    expect(result.length).toBe(returns.length);
    // First windowSize-1 values should be NaN
    for (let i = 0; i < windowSize - 1; i++) {
      expect(Number.isNaN(result[i])).toBe(true);
    }
    // Remaining values should be numbers between -1 and 1
    for (let i = windowSize - 1; i < result.length; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(-1);
      expect(result[i]).toBeLessThanOrEqual(1);
    }
  });

  // Edge cases
  describe('Edge cases', () => {
    it('handles empty arrays', () => {
      expect(correlation([], [])).toBe(0);
      expect(covariance([], [])).toBe(0);
    });

    it('handles single value', () => {
      expect(correlation([0.05], [0.03])).toBe(0);
      expect(covariance([0.05], [0.03])).toBe(0);
    });

    it('handles constant values', () => {
      const constant = [0.01, 0.01, 0.01, 0.01];
      expect(correlation(constant, constant)).toBe(0); // std is 0
    });

    it('handles perfect correlation', () => {
      const x = [0.01, 0.02, 0.03, 0.04];
      const y = [0.02, 0.04, 0.06, 0.08]; // y = 2x
      const result = correlation(x, y);
      assertClose(result, 1, TOLERANCES.TIGHT, 'perfect correlation');
    });

    it('handles perfect negative correlation', () => {
      const x = [0.01, 0.02, 0.03, 0.04];
      const y = [-0.01, -0.02, -0.03, -0.04]; // y = -x
      const result = correlation(x, y);
      assertClose(result, -1, TOLERANCES.TIGHT, 'perfect negative correlation');
    });
  });
});
