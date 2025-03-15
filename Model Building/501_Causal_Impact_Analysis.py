################################################
# Causal Impact Analysis 
################################################

###############################################
# Import required packages
################################################

from causalimpact import CausalImpact
import pandas as pd

################################################
# Import & create data
################################################

# Import data tabels

transations = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
campaign_data = pd.read_excel("data/grocery_database.xlsx", sheet_name = "campaign_data")

# Aggregate transactions data to customer, data Level

customer_daily_sales = transations.groupby(["customer_id", "transaction_date"])["sales_cost"].sum().reset_index()

# Merge on the signup flag

customer_daily_sales = pd.merge( customer_daily_sales, campaign_data, how = "inner", on = "customer_id")


# pivot the data to aggregate daily sales by signup group

causal_impact_df = customer_daily_sales.pivot_table(index = "transaction_date",
                                              columns = "signup_flag",
                                              values = "sales_cost",
                                              aggfunc = "mean")

# provide a frequency for our DateTimeIndex (avoid a warning message)

causal_impact_df.index
"""
DatetimeIndex(['2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04',
               '2020-04-05', '2020-04-06', '2020-04-07', '2020-04-08',
               '2020-04-09', '2020-04-10',
               ...
               '2020-09-21', '2020-09-22', '2020-09-23', '2020-09-24',
               '2020-09-25', '2020-09-26', '2020-09-27', '2020-09-28',
               '2020-09-29', '2020-09-30'],
              dtype='datetime64[ns]', name='transaction_date', length=183, freq=None)
  ---> freq=None can create problem for causal impact algorithm
"""
causal_impact_df.index.freq = "D"

"""
DatetimeIndex(['2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04',
               '2020-04-05', '2020-04-06', '2020-04-07', '2020-04-08',
               '2020-04-09', '2020-04-10',
               ...
               '2020-09-21', '2020-09-22', '2020-09-23', '2020-09-24',
               '2020-09-25', '2020-09-26', '2020-09-27', '2020-09-28',
               '2020-09-29', '2020-09-30'],
              dtype='datetime64[ns]', name='transaction_date', length=183, freq='D')
"""

# for causal impact we need the impacted group in the first column

causal_impact_df = causal_impact_df[[1,0]]

# rename columns to something more meaningful

causal_impact_df.columns = ["member", "non_member"]


################################################
# Apply Causal Impact
################################################

pre_period = ["2020-04-01", "2020-06-30"]
post_period = ["2020-07-01", "2020-09-30"]

ci = CausalImpact(causal_impact_df, pre_period, post_period)


################################################
#Plot the impact
################################################

ci.plot()


################################################
# Extract the summary statistics & report
################################################

print(ci.summary())
"""
Posterior Inference {Causal Impact}
                          Average            Cumulative
Actual                    171.33             15762.67
Prediction (s.d.)         121.42 (4.44)      11170.19 (408.04)
95% CI                    [112.57, 129.96]   [10356.86, 11956.34]

Absolute effect (s.d.)    49.92 (4.44)       4592.48 (408.04)
95% CI                    [41.37, 58.76]     [3806.33, 5405.81]

Relative effect (s.d.)    41.11% (3.65%)     41.11% (3.65%)
95% CI                    [34.08%, 48.39%]   [34.08%, 48.39%]

Posterior tail-area probability p: 0.0
Posterior prob. of a causal effect: 100.0%

For more details run the command: print(impact.summary('report'))
"""

print(ci.summary(output="report"))
"""
Analysis report {CausalImpact}


During the post-intervention period, the response variable had
an average value of approx. 171.33. By contrast, in the absence of an
intervention, we would have expected an average response of 121.42.
The 95% interval of this counterfactual prediction is [112.57, 129.96].
Subtracting this prediction from the observed response yields
an estimate of the causal effect the intervention had on the
response variable. This effect is 49.92 with a 95% interval of
[41.37, 58.76]. For a discussion of the significance of this effect,
see below.


Summing up the individual data points during the post-intervention
period (which can only sometimes be meaningfully interpreted), the
response variable had an overall value of 15762.67.
By contrast, had the intervention not taken place, we would have expected
a sum of 11170.19. The 95% interval of this prediction is [10356.86, 11956.34].


The above results are given in terms of absolute numbers. In relative
terms, the response variable showed an increase of +41.11%. The 95%
interval of this percentage is [34.08%, 48.39%].


This means that the positive effect observed during the intervention
period is statistically significant and unlikely to be due to random
fluctuations. It should be noted, however, that the question of whether
this increase also bears substantive significance can only be answered
by comparing the absolute effect (49.92) to the original goal
of the underlying intervention.


The probability of obtaining this effect by chance is very small
(Bayesian one-sided tail-area probability p = 0.0).
This means the causal effect can be considered statistically
significant.
"""



















