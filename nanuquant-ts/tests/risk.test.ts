/**
 * Tests for risk metrics.
 */

import { describe, it, expect } from 'vitest';
import {
  volatility,
  varMetric,
  cvar,
  maxDrawdown,
  ulcerIndex,
  downsideDeviation,
  toDrawdownSeries,
} from '../src/core/risk';
import {
  loadInputs,
  loadExpected,
  assertClose,
  assertArrayClose,
  normalizeExpectedArray,
  TOLERANCES,
} from './helpers';

interface RiskExpected {
  volatility: number;
  volatilityNotAnnualized: number;
  var95: number;
  var99: number;
  cvar95: number;
  cvar99: number;
  maxDrawdown: number;
  ulcerIndex: number;
  downsideDeviation: number;
  drawdownSeries: (number | null)[];
}

describe('Risk Metrics', () => {
  const inputs = loadInputs();
  const expected = loadExpected<RiskExpected>('risk');
  const returns = inputs.returns;

  it('volatility matches Python (annualized)', () => {
    const result = volatility(returns, { periodsPerYear: 252 });
    assertClose(result, expected.volatility, TOLERANCES.TIGHT, 'volatility');
  });

  it('volatility matches Python (not annualized)', () => {
    const result = volatility(returns, { annualize: false });
    assertClose(result, expected.volatilityNotAnnualized, TOLERANCES.TIGHT, 'volatility not annualized');
  });

  it('var 95% matches Python', () => {
    const result = varMetric(returns, { confidence: 0.95 });
    assertClose(result, expected.var95, TOLERANCES.LOOSE, 'var95');
  });

  it('var 99% matches Python', () => {
    const result = varMetric(returns, { confidence: 0.99 });
    assertClose(result, expected.var99, TOLERANCES.LOOSE, 'var99');
  });

  it('cvar 95% matches Python', () => {
    const result = cvar(returns, { confidence: 0.95 });
    assertClose(result, expected.cvar95, TOLERANCES.LOOSE, 'cvar95');
  });

  it('cvar 99% matches Python', () => {
    const result = cvar(returns, { confidence: 0.99 });
    assertClose(result, expected.cvar99, TOLERANCES.LOOSE, 'cvar99');
  });

  it('maxDrawdown matches Python', () => {
    const result = maxDrawdown(returns);
    assertClose(result, expected.maxDrawdown, TOLERANCES.TIGHT, 'maxDrawdown');
  });

  it('ulcerIndex matches Python', () => {
    const result = ulcerIndex(returns);
    assertClose(result, expected.ulcerIndex, TOLERANCES.TIGHT, 'ulcerIndex');
  });

  it('downsideDeviation matches Python', () => {
    const result = downsideDeviation(returns, { periodsPerYear: 252 });
    assertClose(result, expected.downsideDeviation, TOLERANCES.TIGHT, 'downsideDeviation');
  });

  it('drawdownSeries matches Python', () => {
    const result = toDrawdownSeries(returns);
    const expectedSeries = normalizeExpectedArray(expected.drawdownSeries);
    assertArrayClose(result, expectedSeries, TOLERANCES.TIGHT, 'drawdownSeries');
  });

  // Edge cases
  describe('Edge cases', () => {
    it('handles empty array', () => {
      expect(volatility([])).toBe(0);
      expect(maxDrawdown([])).toBe(0);
      expect(ulcerIndex([])).toBe(0);
    });

    it('handles all positive returns', () => {
      const allPositive = [0.01, 0.02, 0.03, 0.01];
      expect(maxDrawdown(allPositive)).toBe(0);
    });

    it('handles constant returns', () => {
      const constant = [0.01, 0.01, 0.01, 0.01];
      expect(volatility(constant)).toBe(0);
    });
  });
});
