import pandas as pd

questions = pd.read_excel('test/human/data/qa.xlsx', sheet_name='QA', usecols='D')['QUESTION'].tolist()
ground_truths = pd.read_excel('test/human/data/qa.xlsx', sheet_name='QA', usecols='E')['ANSWER'].tolist()