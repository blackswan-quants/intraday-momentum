import pandas as pd
import statsmodels.api as sm

def regressions(daily_returns: pd.Series, factors: pd.DataFrame) -> dict:
    """
    Run OLS regressions of strategy returns against multiple market factors.
    
    Args:
        daily_returns (pd.Series): Daily returns of the strategy.
        factors (pd.DataFrame): DataFrame with factor returns (e.g., market, size, value).
        
    Returns:
        dict: Regression results including alpha, betas, p-values, R-squared metrics.
    """
    # Align the data by concatenation and drop missing values
    data = pd.concat([daily_returns, factors], axis=1).dropna()
    Y = data.iloc[:, 0]       # Strategy returns as dependent variable
    X = sm.add_constant(data.iloc[:, 1:])  # Factor returns + constant term
    
    # Fit the OLS regression model
    model = sm.OLS(Y, X).fit()
    
    # Collect regression statistics in dict
    regression_results = {
        'alpha': model.params['const'],
        'alpha_annualized (%)': model.params['const'] * 252 * 100,
        'alpha_pvalue': model.pvalues['const'],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
    }
    
    # Add factor coefficients (betas) and their p-values
    for factor in factors.columns:
        regression_results[f'beta_{factor}'] = model.params.get(factor, None)
        regression_results[f'beta_{factor}_pvalue'] = model.pvalues.get(factor, None)
    
    return regression_results
