/**
 * Custom exceptions for nanuquant.
 */

/**
 * Base exception for nanuquant.
 */
export class MetricsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MetricsError';
    // Maintains proper stack trace for where error was thrown (only in V8)
    if (typeof (Error as unknown as { captureStackTrace?: unknown }).captureStackTrace === 'function') {
      (Error as unknown as { captureStackTrace: (err: Error, constructor: unknown) => void })
        .captureStackTrace(this, MetricsError);
    }
  }
}

/**
 * Raised when returns series is empty.
 */
export class EmptySeriesError extends MetricsError {
  constructor(message: string = 'Returns series is empty') {
    super(message);
    this.name = 'EmptySeriesError';
  }
}

/**
 * Raised when not enough data for calculation.
 */
export class InsufficientDataError extends MetricsError {
  public readonly required: number;
  public readonly actual: number;
  public readonly metric: string;

  constructor(required: number, actual: number, metric: string = '') {
    const msg = `Insufficient data for ${metric}: requires ${required}, got ${actual}`;
    super(msg);
    this.name = 'InsufficientDataError';
    this.required = required;
    this.actual = actual;
    this.metric = metric;
  }
}

/**
 * Raised when strategy and benchmark lengths differ.
 */
export class BenchmarkMismatchError extends MetricsError {
  public readonly strategyLen: number;
  public readonly benchmarkLen: number;

  constructor(strategyLen: number, benchmarkLen: number) {
    const msg = `Strategy length (${strategyLen}) != benchmark length (${benchmarkLen})`;
    super(msg);
    this.name = 'BenchmarkMismatchError';
    this.strategyLen = strategyLen;
    this.benchmarkLen = benchmarkLen;
  }
}

/**
 * Raised when frequency cannot be determined or is invalid.
 */
export class InvalidFrequencyError extends MetricsError {
  constructor(message: string = 'Cannot determine data frequency') {
    super(message);
    this.name = 'InvalidFrequencyError';
  }
}
