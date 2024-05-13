from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

template = """Question: {question}
        Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.1,
                                                       model_kwargs={"max_length": 64},
                                                       huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN))


question = "Who is the best football team in the United Kingdom?"
llm_result = llm_chain.invoke(question)
print(llm_result)
print(f"\n\nThe answer to the question {llm_result['question']}\n\nis\n\n{llm_result['text']}")


from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, BartConfig

model_id = 'google/flan-t5-base'# go for a smaller model if you dont have capacity on your GPU
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)
print(local_llm.invoke('What is the capital of Germany? '))
llm_chain = LLMChain(prompt=prompt,
                     llm=local_llm
                     )

question = "What is the capital of England?"

print(llm_chain.invoke(question))
