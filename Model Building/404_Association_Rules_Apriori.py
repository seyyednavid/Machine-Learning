################################################
# Association Rule Learning (using Apriori)
################################################

###############################################
# Import required packages
################################################

from apyori import apriori
import pandas as pd

################################################
# Import data
################################################

# Import

alcohol_transactions = pd.read_csv("data/sample_data_apriori.csv")

# Drop unnecessary columns

alcohol_transactions.drop(["transaction_id"], axis = 1, inplace = True)


# Modify data for apriori algorithm 
"""
It does not accept data in the form of a pandas dataframe or an array.
It instead wants us to provide the data as list of lists.
each transaction that we have will be one list and will contain one element for 
each product in that transaction. do not add empty element in list(different list may have different size).
"""
transactions_list = []

for index, row in alcohol_transactions.iterrows():
    transaction = list(row.dropna())
    transactions_list.append(transaction)
    
################################################
# Apply the Apriori algorithm
################################################   
    
apriori_rules = apriori(transactions_list,
                        min_support = 0.003,
                        min_confidence = 0.2,
                        min_lift = 3,
                        min_length = 2,
                        max_length = 2)

apriori_rules = list(apriori_rules)

apriori_rules[0]
"""
 RelationRecord(items=frozenset({'Australian Red'}), support=0.27137650686851694, 
 ordered_statistics=[OrderedStatistic(items_base=frozenset(), 
 items_add=frozenset({'Australian Red'}), confidence=0.27137650686851694, lift=1.0)])
 """
    
################################################
# Convert output to dataframe
################################################    
    
apriori_rules[0][2][0][0]  # frozenset({'American Rose'})  

product1 = [list(rule[2][0][0])[0] for rule in apriori_rules]   
product2 = [list(rule[2][0][1])[0] for rule in apriori_rules] 
support = [rule[1] for rule in apriori_rules]
confidence = [rule[2][0][2] for rule in apriori_rules]
lift = [rule[2][0][3] for rule in apriori_rules]

apriori_rules_df = pd.DataFrame({"product1" : product1,
                                 "product2" : product2,
                                 "support" : support,
                                 "confidence" : confidence,
                                 "lift" : lift})
    
################################################
# Sort rules by descending lift
################################################ 
    
apriori_rules_df.sort_values( by="lift", ascending = False, inplace = True )
    
       
################################################
# Search rules
################################################    

apriori_rules_df[apriori_rules_df["product1"].str.contains("New Zealand")]  
"""
              product1             product2   support  confidence      lift
109    New Zealand Red          Malt Whisky  0.005327    0.271429  5.628987
103    New Zealand Red         Iberia White  0.007289    0.371429  4.616327
111    New Zealand Red    New Zealand White  0.012616    0.642857  4.613826
90     New Zealand Red   French White South  0.004486    0.228571  4.431056
75     New Zealand Red       French White 2  0.009532    0.485714  4.256862
53     New Zealand Red           French Red  0.004205    0.214286  3.879985
63     New Zealand Red     French Red South  0.006448    0.328571  3.868034
113    New Zealand Red        South America  0.010934    0.557143  3.799863
112    New Zealand Red            Other Red  0.004486    0.228571  3.591693
102    New Zealand Red               Iberia  0.012055    0.614286  3.528433
44     New Zealand Red            Champagne  0.008691    0.442857  3.526052
116  New Zealand White  South America White  0.049341    0.354125  3.423206
58     New Zealand Red         French Red 2  0.010093    0.514286  3.359812
114    New Zealand Red  South America White  0.006728    0.342857  3.314286
17     New Zealand Red      Australia White  0.007289    0.371429  3.215742
99     New Zealand Red                  Gin  0.007289    0.371429  3.095527
6      New Zealand Red         American Red  0.006728    0.342857  3.042217
106    New Zealand Red          Italian Red  0.010373    0.528571  3.036094
115  New Zealand White   South Africa White  0.040370    0.289738  3.030783
""" 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


