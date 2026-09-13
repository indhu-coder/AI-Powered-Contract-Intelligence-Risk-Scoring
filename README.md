# AI-Powered-Contract-Intelligence-Risk-Scoring
A sophisticated NLP platform designed for legal and compliance teams.
This system ingests lengthy legal contracts (PDFs/Word docs), automatically extracts key entities (dates, parties,jurisdictions), identifies specific clauses (e.g., termination, confidentiality),and flags anomalous or high-risk language using a fine-tuned Large Language Model (LLM) and
Named Entity Recognition (NER).


Recommended pipeline

      JSON contracts
            ↓
      Python JSON loading
            ↓
      Validate / inspect structure
            ↓
      Convert to DataFrame (for tabular preprocessing)
            ↓
      Text preprocessing
            ↓
      NER / Clause extraction
            ↓
      Embeddings / LLM
            ↓
      Risk scoring
            ↓
      Model / Vector Database



NLP pipeline:
Raw JSON
   ↓
Flatten nested JSON
   ↓
One row per answer
   ↓
Data Cleaning
   ↓
EDA
   ↓
Contract / Clause Text
   ↓
NER
   ↓
Clause Classification
   ↓
Risk Detection
   ↓
Risk Score

Exploratory Data Analysis:

      Data columns (total 9 columns):
       #   Column          Non-Null Count  Dtype  
      ---  ------          --------------  -----  
       0   contract_title  28031 non-null  str    
       1   qa_id           28031 non-null  str    
       2   details         28031 non-null  object 
       3   field           28031 non-null  str    
       4   question        28031 non-null  str    
       5   context_text    28031 non-null  str    
       6   answer_text     28031 non-null  str    
       7   answer_start    13823 non-null  float64
       8   is_impossible   28031 non-null  bool   
      dtypes: bool(1), float64(1), object(1), str(6)
memory usage: 1.7+ MB
None

Shape: (28031, 9)


Null values:

contract_title        0
qa_id                 0
details               0
field                 0
question              0
context_text          0
answer_text           0
answer_start      14208
is_impossible         0


Meaning of the column names:
title = contract
paragraphs = containers for QA records
qas = questions/labels
answers = one or multiple answer spans
answer_start = character position in the original contract text
is_impossible=True = no answer was found for that question

Field Counts:

      Parties                               2555
      License Grant                         1032
      Audit Rights                           939
      Cap On Liability                       907
      Insurance                              904
      Rofr/Rofo/Rofn                         792
      Anti-Assignment                        790
      Post-Termination Services              778
      Minimum Commitment                     769
      Revenue/Profit Sharing                 762
      Exclusivity                            740
      Ip Ownership Assignment                704
      Non-Transferable License               670
      Non-Compete                            650
      Change Of Control                      642
      Warranty Duration                      611
      Irrevocable Or Perpetual License       605
      Volume Restriction                     599
      Covenant Not To Sue                    583
      Joint Ip Ownership                     579
      Termination For Convenience            573
      Liquidated Damages                     570
      Effective Date                         567
      Affiliate License-Licensee             566
      Uncapped Liability                     566
      Expiration Date                        564
      Source Code Escrow                     563
      Competitive Restriction Exception      559
      Affiliate License-Licensor             556
      Renewal Term                           544
      No-Solicit Of Employees                542
      Governing Law                          537
      Non-Disparagement                      537
      No-Solicit Of Customers                534
      Unlimited/All-You-Can-Eat-License      525
      Price Restrictions                     522
      Document Name                          521
      Notice Period To Terminate Renewal     521
      Most Favored Nation                    520
      Third Party Beneficiary                517
      Agreement Date                         516
      Name: count, dtype: int64

Contract Title Counts:
contract_title
Development Agreement                                    1324
STRATEGIC ALLIANCE AGREEMENT                             1304
SPONSORSHIP AGREEMENT                                    1134
Branding Agreement                                       1095
Content License Agreement                                 836
                                                         ... 
Purchase Agreement2                                        43
WEB HOSTING AGREEMENT                                      42
Premium Managed Hosting Agreement                          42
Maintenance and support contract for SICAP(R) modules      42
Franchise Agreement4                                       42
Name: count, Length: 174, dtype: int64