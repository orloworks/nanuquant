/**
 * Global configuration for nanuquant.
 */

import { FrequencyType } from './types';

/**
 * Configuration options for MetricsConfig.
 */
export interface MetricsConfigOptions {
  /** Annualized risk-free rate for excess return calculations. Default: 0.0 */
  riskFreeRate?: number;
  /** Number of trading periods per year for annualization. Default: 252 */
  periodsPerYear?: number;
  /** Expected frequency of return data. Default: "D" */
  frequency?: FrequencyType;
  /** Confidence level for Value at Risk calculations. Default: 0.95 */
  varConfidence?: number;
  /** Default window size for rolling calculations. Default: 252 */
  rollingWindow?: number;
  /** Minimum Acceptable Return for Sortino ratio and related metrics. Default: 0.0 */
  mar?: number;
  /** Delta degrees of freedom for standard deviation calculations. Default: 1 */
  ddof?: number;
}

/**
 * Configuration for metrics calculations.
 */
export class MetricsConfig {
  /** Annualized risk-free rate for excess return calculations. */
  public readonly riskFreeRate: number;
  /** Number of trading periods per year for annualization. */
  public readonly periodsPerYear: number;
  /** Expected frequency of return data. */
  public readonly frequency: FrequencyType;
  /** Confidence level for Value at Risk calculations. */
  public readonly varConfidence: number;
  /** Default window size for rolling calculations. */
  public readonly rollingWindow: number;
  /** Minimum Acceptable Return for Sortino ratio and related metrics. */
  public readonly mar: number;
  /** Delta degrees of freedom for standard deviation calculations. */
  public readonly ddof: number;

  constructor(options: MetricsConfigOptions = {}) {
    this.riskFreeRate = options.riskFreeRate ?? 0.0;
    this.periodsPerYear = options.periodsPerYear ?? 252;
    this.frequency = options.frequency ?? 'D';
    this.varConfidence = options.varConfidence ?? 0.95;
    this.rollingWindow = options.rollingWindow ?? 252;
    this.mar = options.mar ?? 0.0;
    this.ddof = options.ddof ?? 1;
  }
}

/** Global default configuration - users can modify this */
let DEFAULT_CONFIG = new MetricsConfig();

/**
 * Get the current global configuration.
 */
export function getConfig(): MetricsConfig {
  return DEFAULT_CONFIG;
}

/**
 * Set the global configuration.
 * @param config - New configuration to use globally.
 */
export function setConfig(config: MetricsConfig): void {
  DEFAULT_CONFIG = config;
}

/**
 * Reset configuration to defaults.
 */
export function resetConfig(): void {
  DEFAULT_CONFIG = new MetricsConfig();
}
