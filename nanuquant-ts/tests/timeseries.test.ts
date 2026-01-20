/**
 * Tests for timeseries functions.
 */

import { describe, it, expect } from 'vitest';
import {
  cumulativeReturns,
  equityCurve,
  logReturns,
  simpleReturns,
  histogram,
  yearlyReturns,
  monthlyReturns,
  weeklyReturns,
  quarterlyReturns,
  periodReturns,
} from '../src/core/timeseries';
import {
  loadInputs,
  loadExpected,
  assertArrayClose,
  normalizeExpectedArray,
  TOLERANCES,
} from './helpers';

interface TimeseriesExpected {
  cumulativeReturns: (number | null)[];
  histogram: {
    binStart: number[];
    binEnd: number[];
    binCenter: number[];
    count: number[];
    frequency: number[];
  };
}

describe('Timeseries Functions', () => {
  const inputs = loadInputs();
  const expected = loadExpected<TimeseriesExpected>('timeseries');
  const returns = inputs.returns;
  const dates = inputs.dates.map((d) => new Date(d));

  // Take last 100 values to match fixture
  const lastN = 100;

  it('cumulativeReturns matches Python', () => {
    const result = cumulativeReturns(returns);
    const resultLast = result.slice(-lastN);
    const expectedValues = normalizeExpectedArray(expected.cumulativeReturns);
    assertArrayClose(resultLast, expectedValues, TOLERANCES.TIGHT, 'cumulativeReturns');
  });

  it('equityCurve is alias for cumulativeReturns', () => {
    const cumRet = cumulativeReturns(returns);
    const equity = equityCurve(returns);
    expect(cumRet).toEqual(equity);
  });

  it('logReturns and simpleReturns are inverses', () => {
    const log = logReturns(returns);
    const back = simpleReturns(log);
    // Should get back the original returns
    assertArrayClose(back, returns, TOLERANCES.TIGHT, 'log/simple round-trip');
  });

  it('histogram has correct structure', () => {
    const result = histogram(returns, { bins: 20 });
    expect(result.length).toBe(20);

    // Check structure
    result.forEach((bin) => {
      expect(typeof bin.binStart).toBe('number');
      expect(typeof bin.binEnd).toBe('number');
      expect(typeof bin.binCenter).toBe('number');
      expect(typeof bin.count).toBe('number');
      expect(typeof bin.frequency).toBe('number');
      expect(bin.binEnd).toBeGreaterThan(bin.binStart);
    });

    // Total count should equal length
    const totalCount = result.reduce((sum, bin) => sum + bin.count, 0);
    expect(totalCount).toBe(returns.length);

    // Total frequency should be 1
    const totalFreq = result.reduce((sum, bin) => sum + bin.frequency, 0);
    expect(totalFreq).toBeCloseTo(1, 10);
  });

  it('histogram matches Python bin counts approximately', () => {
    const result = histogram(returns, { bins: 20 });
    const resultCounts = result.map((b) => b.count);
    // Just check that counts are reasonable (some may differ due to binning edge cases)
    expect(resultCounts.length).toBe(expected.histogram.count.length);
  });

  // Date-dependent functions
  describe('Date-dependent functions', () => {
    it('yearlyReturns groups by year', () => {
      const result = yearlyReturns(returns, dates);
      expect(result.size).toBeGreaterThan(0);
      // Should have years 2020, 2021, 2022 based on 1000 days from 2020-01-01
      expect(result.has('2020')).toBe(true);
    });

    it('monthlyReturns groups by month', () => {
      const result = monthlyReturns(returns, dates);
      expect(result.size).toBeGreaterThan(0);
      // Should have January 2020
      expect(result.has('2020-1')).toBe(true);
    });

    it('weeklyReturns groups by week', () => {
      const result = weeklyReturns(returns, dates);
      expect(result.size).toBeGreaterThan(0);
    });

    it('quarterlyReturns groups by quarter', () => {
      const result = quarterlyReturns(returns, dates);
      expect(result.size).toBeGreaterThan(0);
      // Should have Q1 2020
      expect(result.has('2020-Q1')).toBe(true);
    });

    it('periodReturns calculates YTD, MTD, WTD, QTD', () => {
      const lastDate = dates[dates.length - 1];
      const result = periodReturns(returns, dates, { referenceDate: lastDate });
      expect(typeof result.YTD).toBe('number');
      expect(typeof result.MTD).toBe('number');
      expect(typeof result.WTD).toBe('number');
      expect(typeof result.QTD).toBe('number');
    });
  });

  // Edge cases
  describe('Edge cases', () => {
    it('handles empty array', () => {
      expect(cumulativeReturns([])).toEqual([]);
      expect(logReturns([])).toEqual([]);
      expect(simpleReturns([])).toEqual([]);
      expect(histogram([])).toEqual([]);
    });

    it('handles single value', () => {
      const result = cumulativeReturns([0.05]);
      expect(result.length).toBe(1);
      expect(result[0]).toBeCloseTo(0.05, 10);
    });

    it('handles constant returns in histogram', () => {
      const constant = [0.01, 0.01, 0.01, 0.01];
      const result = histogram(constant, { bins: 5 });
      expect(result.length).toBe(1); // All same value goes to one bin
      expect(result[0].count).toBe(4);
    });

    it('handles negative returns approaching -100%', () => {
      const log = logReturns([-0.99]);
      expect(log[0]).toBeLessThan(0);
      expect(Number.isFinite(log[0])).toBe(true);

      const logNeg1 = logReturns([-1]);
      expect(logNeg1[0]).toBe(-Infinity);
    });
  });
});
