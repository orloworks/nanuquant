# Legal Disclaimer and Important Notices

## NOT FINANCIAL ADVICE

**NanuQuant is a software library for quantitative analysis and educational purposes only. Nothing in this library, its documentation, examples, or outputs constitutes financial advice, investment advice, trading advice, or any other form of professional advice.**

The metrics, calculations, analyses, and any other outputs produced by NanuQuant:

- Are provided for **informational and educational purposes only**
- Should **NOT** be used as the sole basis for any investment decisions
- Do **NOT** constitute recommendations to buy, sell, or hold any securities, derivatives, or other financial instruments
- Are **NOT** a substitute for professional financial advice from a qualified financial advisor, broker, or other licensed professional

## No Warranty

NanuQuant is provided "AS IS" without warranty of any kind, express or implied, including but not limited to:

- **Accuracy**: We do not warrant that calculations are error-free or that results will meet your requirements
- **Completeness**: The library may not include all metrics or calculations relevant to your specific use case
- **Fitness for Purpose**: We make no representations about the suitability of this software for any particular purpose
- **Timeliness**: Results reflect the data provided and may not account for current market conditions

## Limitation of Liability

**IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS, OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

This includes, but is not limited to:

- Direct, indirect, incidental, special, consequential, or punitive damages
- Loss of profits, revenue, data, or use
- Business interruption
- Any financial losses resulting from investment decisions made using this software

## Assumptions and Limitations

### Statistical Assumptions

Many metrics in NanuQuant rely on statistical assumptions that may not hold in real financial markets:

1. **Normal Distribution**: Metrics like parametric VaR assume normally distributed returns. Real market returns often exhibit fat tails, skewness, and other non-normal characteristics.

2. **Stationarity**: Many calculations assume returns are stationary (stable statistical properties over time). Markets experience regime changes, structural breaks, and varying volatility.

3. **Independence**: Some metrics assume returns are independent. In reality, financial returns often exhibit autocorrelation and volatility clustering.

4. **Historical Data**: Past performance is NOT indicative of future results. Backtested strategies may not perform similarly in live trading due to:
   - Survivorship bias
   - Look-ahead bias
   - Overfitting to historical data
   - Market impact not captured in historical data
   - Changing market regimes

### Practical Limitations

1. **Data Quality**: Results are only as good as the input data. Errors, gaps, or inconsistencies in data will affect calculations.

2. **Transaction Costs**: Unless explicitly modeled, metrics do not account for:
   - Commissions and fees
   - Bid-ask spreads
   - Slippage
   - Market impact
   - Financing costs

3. **Taxes**: Tax implications are not considered in any calculations.

4. **Liquidity**: Metrics assume positions can be entered and exited at observed prices, which may not be true for:
   - Large positions
   - Illiquid securities
   - Fast-moving markets

## Appropriate Use Cases

NanuQuant is designed for:

- **Educational purposes**: Learning about quantitative finance concepts
- **Research**: Academic and professional research analysis
- **Strategy development**: Initial screening and analysis of trading ideas
- **Risk monitoring**: Understanding portfolio risk characteristics
- **Performance attribution**: Analyzing historical performance

NanuQuant is **NOT** designed for:

- Automated trading without human oversight
- Sole basis for investment decisions
- Regulatory compliance calculations (consult appropriate regulatory guidance)
- High-frequency trading systems (verify latency requirements independently)

## Professional Advice Recommended

Before making any investment decisions, we strongly recommend:

1. **Consult a qualified financial advisor** who understands your specific financial situation, goals, and risk tolerance
2. **Conduct your own due diligence** and research
3. **Understand the risks** associated with any investment strategy
4. **Verify all calculations independently** before relying on them
5. **Paper trade** any strategy before committing real capital

## Regulatory Notice

Users are responsible for ensuring their use of NanuQuant complies with all applicable laws, regulations, and guidelines in their jurisdiction, including but not limited to:

- Securities regulations
- Investment advisor regulations
- Data privacy laws
- Financial reporting requirements

## Acknowledgment

By using NanuQuant, you acknowledge that:

1. You have read and understood this disclaimer
2. You accept all risks associated with using this software
3. You will not hold the authors or contributors liable for any losses
4. You understand that past performance does not guarantee future results
5. You will seek professional advice before making investment decisions

---

**If you do not agree with these terms, do not use NanuQuant.**

---

*Last updated: January 2025*
