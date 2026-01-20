/**
 * Tests for rolling metrics.
 */

import { describe, it, expect } from 'vitest';
import {
  rollingVolatility,
  rollingSharpe,
  rollingSortino,
  rollingBeta,
  rollingGreeks,
} from '../src/core/rolling';
import {
  loadInputs,
  loadExpected,
  assertArrayClose,
  normalizeExpectedArray,
  TOLERANCES,
} from './helpers';

interface RollingExpected {
  rollingVolatility: (number | null)[];
  rollingSharpe: (number | null)[];
  rollingSortino: (number | null)[];
  rollingBeta: (number | null)[];
  rollingAlpha: (number | null)[];
  rollingBetaGreeks: (number | null)[];
}

describe('Rolling Metrics', () => {
  const inputs = loadInputs();
  const expected = loadExpected<RollingExpected>('rolling');
  const returns = inputs.returns;
  const benchmark = inputs.benchmark;

  // Take last 100 values to match fixture
  const rollingPeriod = 126;
  const lastN = 100;

  it('rollingVolatility matches Python', () => {
    const result = rollingVolatility(returns, { rollingPeriod });
    const resultLast = result.slice(-lastN);
    const expectedValues = normalizeExpectedArray(expected.rollingVolatility);
    assertArrayClose(resultLast, expectedValues, TOLERANCES.LOOSE, 'rollingVolatility');
  });

  it('rollingSharpe matches Python', () => {
    const result = rollingSharpe(returns, { rollingPeriod });
    const resultLast = result.slice(-lastN);
    const expectedValues = normalizeExpectedArray(expected.rollingSharpe);
    assertArrayClose(resultLast, expectedValues, TOLERANCES.LOOSE, 'rollingSharpe');
  });

  it('rollingSortino matches Python', () => {
    const result = rollingSortino(returns, { rollingPeriod });
    const resultLast = result.slice(-lastN);
    const expectedValues = normalizeExpectedArray(expected.rollingSortino);
    // Use STAT tolerance for Sortino as it can vary more
    assertArrayClose(resultLast, expectedValues, TOLERANCES.STAT, 'rollingSortino');
  });

  it('rollingBeta matches Python', () => {
    const result = rollingBeta(returns, benchmark, { rollingPeriod });
    const resultLast = result.slice(-lastN);
    const expectedValues = normalizeExpectedArray(expected.rollingBeta);
    assertArrayClose(resultLast, expectedValues, TOLERANCES.LOOSE, 'rollingBeta');
  });

  it('rollingGreeks matches Python', () => {
    const result = rollingGreeks(returns, benchmark, { rollingPeriod });
    const alphaLast = result.alpha.slice(-lastN);
    const betaLast = result.beta.slice(-lastN);

    const expectedAlpha = normalizeExpectedArray(expected.rollingAlpha);
    const expectedBeta = normalizeExpectedArray(expected.rollingBetaGreeks);

    assertArrayClose(alphaLast, expectedAlpha, TOLERANCES.LOOSE, 'rollingGreeks.alpha');
    assertArrayClose(betaLast, expectedBeta, TOLERANCES.LOOSE, 'rollingGreeks.beta');
  });

  // Edge cases
  describe('Edge cases', () => {
    it('returns empty for insufficient data', () => {
      const short = [0.01, 0.02, 0.03];
      expect(rollingVolatility(short, { rollingPeriod: 126 })).toEqual([]);
      expect(rollingSharpe(short, { rollingPeriod: 126 })).toEqual([]);
    });

    it('returns correct length', () => {
      const period = 10;
      const result = rollingVolatility(returns, { rollingPeriod: period });
      expect(result.length).toBe(returns.length);
    });

    it('first rollingPeriod-1 values are NaN', () => {
      const period = 10;
      const result = rollingVolatility(returns, { rollingPeriod: period });
      for (let i = 0; i < period - 1; i++) {
        expect(Number.isNaN(result[i])).toBe(true);
      }
      // First valid value should be at index period-1
      expect(Number.isFinite(result[period - 1])).toBe(true);
    });
  });
});
