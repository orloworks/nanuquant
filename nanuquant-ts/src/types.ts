/**
 * Type definitions for nanuquant.
 */

/**
 * Supported frequency types for return data.
 * D=daily, W=weekly, M=monthly, H=hourly, min=minute, s=second
 */
export type FrequencyType = 'D' | 'W' | 'M' | 'H' | 'min' | 's';

/**
 * Annualization periods for each frequency.
 * D=daily (252 trading days), W=weekly, M=monthly, H=hourly (6.5h/day), min=minute
 */
export const ANNUALIZATION_PERIODS: Record<FrequencyType, number> = {
  D: 252.0,
  W: 52.0,
  M: 12.0,
  H: 252.0 * 6.5, // 1638 trading hours/year
  min: 252.0 * 390.0, // 98280 trading minutes/year
  s: 252.0 * 390.0 * 60.0, // trading seconds/year
};

/**
 * Standard deviation multipliers for parametric VaR at common confidence levels.
 */
export const VAR_SIGMA_MAP: Record<number, number> = {
  0.9: 1.282,
  0.95: 1.645,
  0.99: 2.326,
};

/**
 * Rolling window preset types.
 */
export type RollingWindowType = '1M' | '3M' | '6M' | '1Y' | '2Y' | '3Y' | '5Y';

/**
 * Rolling window presets in trading days.
 */
export const ROLLING_WINDOW_DAYS: Record<RollingWindowType, number> = {
  '1M': 21,
  '3M': 63,
  '6M': 126,
  '1Y': 252,
  '2Y': 504,
  '3Y': 756,
  '5Y': 1260,
};

/**
 * Greek metrics return type.
 */
export interface Greeks {
  alpha: number;
  beta: number;
}

/**
 * Rolling greeks return type.
 */
export interface RollingGreeks {
  alpha: number[];
  beta: number[];
}
