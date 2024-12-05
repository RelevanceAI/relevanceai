from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

client.beta.assistants.create()


