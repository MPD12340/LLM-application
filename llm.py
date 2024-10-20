from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from decouple import config

HUGGING_FACE_API_TOKEN = config("HUGGING_FACE_API_TOKEN")

template = "<s>[INST] {question} [/INST]"

prompt_template = PromptTemplate.from_template(template)
ask_input = input(" Ask a question ?")
formatted_prompt_template = prompt_template.format(question=ask_input)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, huggingfacehub_api_token=HUGGING_FACE_API_TOKEN
)

response = llm.invoke(formatted_prompt_template)
print(response)
