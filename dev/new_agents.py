import sys
sys.path.append(".")

import time
from dotenv import load_dotenv
load_dotenv()

from relevanceai import RelevanceAI

client = RelevanceAI()

my_agents = client.agents.list_agents()

company_research_agent = client.agents.retrieve_agent("b9cd9cd5-2aea-45dd-b12a-461a0935528e")
print(company_research_agent)