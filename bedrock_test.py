#from langchain_community.llms.bedrock import Bedrock
from langchain.chains import LLMChain

from langchain.prompts import PromptTemplate
from langchain_aws import BedrockLLM
import boto3


#bedrock client
bedrock_client = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-east-1"
)

model_id = "ai21.j2-mid-v1"

llm = BedrockLLM(
    model_id = model_id,
    client = bedrock_client
)


def my_model(user_prompt):
    prompt = PromptTemplate(
        input_variables=['user_prompt'],
        template= "You are a chatbot. provide ans for {user_prompt}"
    )

    bedrock_chain = LLMChain(llm = llm, prompt = prompt)
    response = bedrock_chain({'user_prompt': user_prompt})

    return response



user_prompt = "What is Python?"
res = my_model(user_prompt)
print(res['text'])