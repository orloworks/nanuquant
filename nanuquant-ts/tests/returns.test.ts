/**
 * Tests for returns metrics.
 */

import { describe, it, expect } from 'vitest';
import {
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
} from '../src/core/returns';
import { loadInputs, loadExpected, assertClose, TOLERANCES } from './helpers';

interface ReturnsExpected {
  comp: number;
  cagr: number;
  avgReturn: number;
  avgWin: number;
  avgLoss: number;
  best: number;
  worst: number;
  winRate: number;
  payoffRatio: number;
  profitFactor: number;
  consecutiveWins: number;
  consecutiveLosses: number;
}

describe('Returns Metrics', () => {
  const inputs = loadInputs();
  const expected = loadExpected<ReturnsExpected>('returns');
  const returns = inputs.returns;

  it('comp matches Python', () => {
    const result = comp(returns);
    assertClose(result, expected.comp, TOLERANCES.TIGHT, 'comp');
  });

  it('cagr matches Python', () => {
    const result = cagr(returns, { periodsPerYear: 252 });
    assertClose(result, expected.cagr, TOLERANCES.TIGHT, 'cagr');
  });

  it('avgReturn matches Python', () => {
    const result = avgReturn(returns);
    assertClose(result, expected.avgReturn, TOLERANCES.TIGHT, 'avgReturn');
  });

  it('avgWin matches Python', () => {
    const result = avgWin(returns);
    assertClose(result, expected.avgWin, TOLERANCES.TIGHT, 'avgWin');
  });

  it('avgLoss matches Python', () => {
    const result = avgLoss(returns);
    assertClose(result, expected.avgLoss, TOLERANCES.TIGHT, 'avgLoss');
  });

  it('best matches Python', () => {
    const result = best(returns);
    assertClose(result, expected.best, TOLERANCES.TIGHT, 'best');
  });

  it('worst matches Python', () => {
    const result = worst(returns);
    assertClose(result, expected.worst, TOLERANCES.TIGHT, 'worst');
  });

  it('winRate matches Python', () => {
    const result = winRate(returns);
    assertClose(result, expected.winRate, TOLERANCES.TIGHT, 'winRate');
  });

  it('payoffRatio matches Python', () => {
    const result = payoffRatio(returns);
    assertClose(result, expected.payoffRatio, TOLERANCES.TIGHT, 'payoffRatio');
  });

  it('profitFactor matches Python', () => {
    const result = profitFactor(returns);
    assertClose(result, expected.profitFactor, TOLERANCES.TIGHT, 'profitFactor');
  });

  it('consecutiveWins matches Python', () => {
    const result = consecutiveWins(returns);
    expect(result).toBe(expected.consecutiveWins);
  });

  it('consecutiveLosses matches Python', () => {
    const result = consecutiveLosses(returns);
    expect(result).toBe(expected.consecutiveLosses);
  });

  // Edge cases
  describe('Edge cases', () => {
    it('handles empty array', () => {
      expect(comp([])).toBe(0);
      expect(cagr([])).toBe(0);
      expect(avgReturn([])).toBe(0);
      expect(winRate([])).toBe(0);
    });

    it('handles single value', () => {
      expect(comp([0.05])).toBeCloseTo(0.05, 10);
      expect(avgReturn([0.05])).toBeCloseTo(0.05, 10);
    });

    it('handles all positive returns', () => {
      const allPositive = [0.01, 0.02, 0.03, 0.01];
      expect(winRate(allPositive)).toBe(1);
      expect(avgLoss(allPositive)).toBe(0);
    });

    it('handles all negative returns', () => {
      const allNegative = [-0.01, -0.02, -0.03, -0.01];
      expect(winRate(allNegative)).toBe(0);
      expect(avgWin(allNegative)).toBe(0);
    });
  });
});
