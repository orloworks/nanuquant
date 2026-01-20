/**
 * Tests for performance metrics.
 */

import { describe, it, expect } from 'vitest';
import {
  sharpe,
  sortino,
  calmar,
  omega,
  gainToPainRatio,
  ulcerPerformanceIndex,
  kellyCriterion,
  tailRatio,
  commonSenseRatio,
  riskReturnRatio,
  recoveryFactor,
  greeks,
  informationRatio,
  rSquared,
  treynorRatio,
  benchmarkCorrelation,
} from '../src/core/performance';
import { loadInputs, loadExpected, assertClose, TOLERANCES } from './helpers';

interface PerformanceExpected {
  sharpe: number;
  sharpeRf: number;
  sortino: number;
  calmar: number;
  omega: number;
  gainToPainRatio: number;
  ulcerPerformanceIndex: number;
  kellyCriterion: number;
  tailRatio: number;
  commonSenseRatio: number;
  riskReturnRatio: number;
  recoveryFactor: number;
  greeks: { alpha: number; beta: number };
  informationRatio: number;
  rSquared: number;
  treynorRatio: number;
  benchmarkCorrelation: number;
}

describe('Performance Metrics', () => {
  const inputs = loadInputs();
  const expected = loadExpected<PerformanceExpected>('performance');
  const returns = inputs.returns;
  const benchmark = inputs.benchmark;

  it('sharpe matches Python', () => {
    const result = sharpe(returns, { periodsPerYear: 252 });
    assertClose(result, expected.sharpe, TOLERANCES.TIGHT, 'sharpe');
  });

  it('sharpe with risk-free rate matches Python', () => {
    const result = sharpe(returns, { riskFreeRate: 0.02, periodsPerYear: 252 });
    assertClose(result, expected.sharpeRf, TOLERANCES.TIGHT, 'sharpe with rf');
  });

  it('sortino matches Python', () => {
    const result = sortino(returns, { periodsPerYear: 252 });
    assertClose(result, expected.sortino, TOLERANCES.TIGHT, 'sortino');
  });

  it('calmar matches Python', () => {
    const result = calmar(returns, { periodsPerYear: 252 });
    assertClose(result, expected.calmar, TOLERANCES.TIGHT, 'calmar');
  });

  it('omega matches Python', () => {
    const result = omega(returns);
    assertClose(result, expected.omega, TOLERANCES.TIGHT, 'omega');
  });

  it('gainToPainRatio matches Python', () => {
    const result = gainToPainRatio(returns);
    assertClose(result, expected.gainToPainRatio, TOLERANCES.TIGHT, 'gainToPainRatio');
  });

  it('ulcerPerformanceIndex matches Python', () => {
    const result = ulcerPerformanceIndex(returns);
    assertClose(result, expected.ulcerPerformanceIndex, TOLERANCES.TIGHT, 'ulcerPerformanceIndex');
  });

  it('kellyCriterion matches Python', () => {
    const result = kellyCriterion(returns);
    assertClose(result, expected.kellyCriterion, TOLERANCES.LOOSE, 'kellyCriterion');
  });

  it('tailRatio matches Python', () => {
    const result = tailRatio(returns);
    assertClose(result, expected.tailRatio, TOLERANCES.TIGHT, 'tailRatio');
  });

  it('commonSenseRatio matches Python', () => {
    const result = commonSenseRatio(returns);
    assertClose(result, expected.commonSenseRatio, TOLERANCES.TIGHT, 'commonSenseRatio');
  });

  it('riskReturnRatio matches Python', () => {
    const result = riskReturnRatio(returns);
    assertClose(result, expected.riskReturnRatio, TOLERANCES.TIGHT, 'riskReturnRatio');
  });

  it('recoveryFactor matches Python', () => {
    const result = recoveryFactor(returns);
    assertClose(result, expected.recoveryFactor, TOLERANCES.TIGHT, 'recoveryFactor');
  });

  it('greeks matches Python', () => {
    const result = greeks(returns, benchmark);
    assertClose(result.alpha, expected.greeks.alpha, TOLERANCES.TIGHT, 'greeks.alpha');
    assertClose(result.beta, expected.greeks.beta, TOLERANCES.TIGHT, 'greeks.beta');
  });

  it('informationRatio matches Python', () => {
    const result = informationRatio(returns, benchmark);
    assertClose(result, expected.informationRatio, TOLERANCES.TIGHT, 'informationRatio');
  });

  it('rSquared matches Python', () => {
    const result = rSquared(returns, benchmark);
    assertClose(result, expected.rSquared, TOLERANCES.TIGHT, 'rSquared');
  });

  it('treynorRatio matches Python', () => {
    const result = treynorRatio(returns, benchmark, { periodsPerYear: 252 });
    assertClose(result, expected.treynorRatio, TOLERANCES.TIGHT, 'treynorRatio');
  });

  it('benchmarkCorrelation matches Python', () => {
    const result = benchmarkCorrelation(returns, benchmark);
    assertClose(result, expected.benchmarkCorrelation, TOLERANCES.TIGHT, 'benchmarkCorrelation');
  });

  // Edge cases
  describe('Edge cases', () => {
    it('handles empty array', () => {
      expect(sharpe([])).toBe(0);
      expect(sortino([])).toBe(0);
      // omega returns NaN for insufficient data (< 2 points)
      expect(Number.isNaN(omega([]))).toBe(true);
    });

    it('handles all positive returns for omega', () => {
      const allPositive = [0.01, 0.02, 0.03, 0.01];
      // All positive returns with default threshold=0 means no losses, returns Infinity
      expect(omega(allPositive)).toBe(Infinity);
    });
  });
});
