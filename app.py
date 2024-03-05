import json

import os
import openai
import pandas as pd
from langchain_experimental.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain_openai.llms import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentType, AgentExecutor
import warnings
from datetime import datetime
import streamlit as st
from PIL import Image
import shutil
#import hydralit_components as hc



now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

OPENAI_API_KEY = "6681d53111984885810f85d82c9ab8e7"
OPENAI_DEPLOYMENT_NAME = "gpt-4"
MODEL_NAME = "gpt-4"
AZURE_ENDPOINT = "https://gpt4-ibm4.openai.azure.com/"
API_VERSION = "2023-07-01-preview"

llm =  AzureChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    deployment_name=OPENAI_DEPLOYMENT_NAME,
    model_name=MODEL_NAME, 
    api_version=API_VERSION,
    azure_endpoint = AZURE_ENDPOINT,
    temperature=0,
    timeout=600,
    verbose=True,
    max_retries=1
    
)
warnings.filterwarnings("ignore")
os.system('cls')
print('===================='+dt_string+'===========================================')


mypath = ".\chart_image" 
for root, dirs, files in os.walk(mypath, topdown=False):
    for file in files:
        os.remove(os.path.join(root, file))

st.title("🤔 Chat with your CSV 📊")
st.write("Please upload your CSV and metadata file below.")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='windows-1252')

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Query:", placeholder="Enter your query here.", key='input')
        submit_button = st.form_submit_button(label='Send')

        chart_here= '{"chart": "Here is the chart:"}'
        prompt = f"""
        {user_input}, create bar chart.
        You must need to use matplotlib library if required to create a any chart.
        If the query requires creating a chart, please save the chart as "./chart_image/chart.png" and "Here is the chart:" when reply as follows:
        {chart_here}
        """

        prompt_insight = f"""
        {user_input}, Summarize and provide data insight, product in details with product name and the result in text.
        Follow the below example and do not use example data.

        Example:

        ******Products:******
        1. AT&T CL83451 4-Handset Telephone: 


        ****Insights:*****
        Provide insight here for the products with the product name in bullet number, recomendation to increase sales.
        Insight example:
        -Bush Somerset Collection Bookcase: This product has a sales of 261.96, with a quantity of 2, no discount, and a profit of 41.9136. To increase sales, consider offering a small discount or bundling this bookcase with other office furniture items.
        -Hon Deluxe Fabric Upholstered Stacking Chairs, Rounded Back: This product has a sales of 731.94, with a quantity of 3, no discount, and a profit of 219.582. To increase sales, consider offering a discount on bulk purchases or promoting the comfort and durability of these chairs.
        """

        table_here= '{"table": "Here is the table:"}'
        table = '{"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}"}'

        prompt_table = f"""
        Lets think step by step.
        Here is the query: 
        {user_input}
        If the query requires creating a table, reply as follows:
        {table}, plot this table here.
        """

        print('prompt')
        print(prompt)

        agent_df = create_pandas_dataframe_agent(llm, df, verbose=True)
        agent_df.handle_parsing_errors=True
        try:
            if submit_button and user_input:
                result_df = agent_df.run(prompt)
                print('----data frame output-----')
                print(result_df)
                st.write(result_df)
       
                filepath = '.\chart_image\chart.png'
                if os.path.isfile(filepath):
                    image = Image.open(filepath)
                    st.image(image, caption='Enter any caption here')
                else:
                     print('File not exists')



                result_insight_df = agent_df.run(prompt_insight)
                st.write('--------------')
                st.write(result_insight_df)

                result_table_df = agent_df.run(prompt_table)
                st.write('--------------')
                st.write(result_table_df)
                st.table(result_table_df)

        except ValueError as e:
            print(f'Caught a ValueError: {e}')


        
