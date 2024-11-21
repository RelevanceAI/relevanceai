import sys
sys.path.append(".")
from relevanceai import RelevanceAI
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv()

client = RelevanceAI()

agent_id = "b9cd9cd5-2aea-45dd-b12a-461a0935528e"

message = dedent(f"""
Research the following company:
                 
Vividly govividly.com      
""")

task = client.tasks.trigger_task(
    agent_id=agent_id,
    message=message,
    override=True,
    debug_mode_config_id={
        'tool_configs': {
            "overrides": {
                "61ab4970891d9b0d": {
                    "input_overrides_enabled": True,
                    "input_overrides": {
                        "company_linkedin_url": "https://au.linkedin.com/company/relevanceai",
                        "product_context": """Relevance AI is a 3 year old award-winning AI company that developed a no-code AI platform that allows users to create ther own, powerful AI Tools within minutes, without writing any code. 

ChatGPT and other large languege models lack security and may expose your data to other users as it trains itself with it. It isn’t trained on your personal data, has a character limit, and it’s difficult to reuse your prompts. It's also quite manual, as you can't automate processes with it.

We've created a user-friendly solution to all these challenges.

These AI apps you can create with Relevance AI can automate the human-dependent aspects of almost *any* process.

Using this solution, you can combine Large Language Models and any other AI technologies on your own data.

We're stepping into the era of customizable, autonomous AI agents for task automation, without needing any coding knowledge. This can increase the productivity of businesses by 50x. 

ChatGPT and other large languege models lack security and may expose your data to other users as it trains itself with it. It isn’t trained on your personal data, has a character limit, and it’s difficult to reuse your prompts. It's also quite manual, as you can't automate processes with it.

We've created a user-friendly solution to all these challenges."""
                    },
                }
            }
        }
    }
)

print(task)
