/**
 * Tests for distribution metrics.
 */

import { describe, it, expect } from 'vitest';
import {
  skewness,
  kurtosis,
  expectedReturn,
  geometricMean,
  outlierWinRatio,
  outlierLossRatio,
  outliers,
  removeOutliers,
} from '../src/core/distribution';
import { loadInputs, loadExpected, assertClose, TOLERANCES } from './helpers';

interface DistributionExpected {
  skewness: number;
  kurtosis: number;
  expectedReturn: number;
  geometricMean: number;
  outlierWinRatio: number;
  outlierLossRatio: number;
}

describe('Distribution Metrics', () => {
  const inputs = loadInputs();
  const expected = loadExpected<DistributionExpected>('distribution');
  const returns = inputs.returns;

  it('skewness matches Python', () => {
    const result = skewness(returns);
    // Use STATISTICAL tolerance because different libraries use different bias correction
    assertClose(result, expected.skewness, TOLERANCES.STATISTICAL, 'skewness');
  });

  it('kurtosis matches Python', () => {
    const result = kurtosis(returns);
    // Use STATISTICAL tolerance because different libraries use different bias correction
    assertClose(result, expected.kurtosis, TOLERANCES.STATISTICAL, 'kurtosis');
  });

  it('expectedReturn matches Python', () => {
    const result = expectedReturn(returns);
    assertClose(result, expected.expectedReturn, TOLERANCES.TIGHT, 'expectedReturn');
  });

  it('geometricMean matches Python', () => {
    const result = geometricMean(returns);
    assertClose(result, expected.geometricMean, TOLERANCES.TIGHT, 'geometricMean');
  });

  it('outlierWinRatio matches Python', () => {
    const result = outlierWinRatio(returns);
    assertClose(result, expected.outlierWinRatio, TOLERANCES.LOOSE, 'outlierWinRatio');
  });

  it('outlierLossRatio matches Python', () => {
    const result = outlierLossRatio(returns);
    assertClose(result, expected.outlierLossRatio, TOLERANCES.LOOSE, 'outlierLossRatio');
  });

  it('outliers returns array of outliers', () => {
    const result = outliers(returns);
    expect(Array.isArray(result)).toBe(true);
    // All outliers should be numbers
    result.forEach((v) => expect(typeof v).toBe('number'));
  });

  it('removeOutliers returns filtered array', () => {
    const result = removeOutliers(returns);
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBeLessThanOrEqual(returns.length);
  });

  // Edge cases
  describe('Edge cases', () => {
    it('handles empty array', () => {
      expect(skewness([])).toBe(0);
      expect(kurtosis([])).toBe(0);
      expect(expectedReturn([])).toBe(0);
      expect(geometricMean([])).toBe(0);
    });

    it('handles insufficient data for skewness', () => {
      expect(skewness([0.01, 0.02])).toBe(0); // Need at least 3 points
    });

    it('handles insufficient data for kurtosis', () => {
      expect(kurtosis([0.01, 0.02, 0.03])).toBe(0); // Need at least 4 points
    });

    it('handles symmetric distribution', () => {
      const symmetric = [0.01, 0.02, 0.03, 0.02, 0.01];
      const sk = skewness(symmetric);
      // Symmetric distribution should have skewness close to 0
      expect(Math.abs(sk)).toBeLessThan(1);
    });

    it('handles outliers correctly', () => {
      const withOutliers = [0.01, 0.02, 0.01, 0.02, 0.5, -0.5]; // 0.5 and -0.5 are outliers
      const filtered = removeOutliers(withOutliers);
      expect(filtered.length).toBeLessThan(withOutliers.length);
    });
  });
});
