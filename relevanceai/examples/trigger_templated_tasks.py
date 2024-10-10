
"""
Trigger templated tasks. 
"""

import time 
import csv
import time 
from textwrap import dedent

from relevanceai import RelevanceAI
from dotenv import load_dotenv
load_dotenv()

client = RelevanceAI()

agent_id = "xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

path = "test_leads.csv"

with open(path, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    for i, r in enumerate(reader):

        print(f"Row {i}...")

        row = {
            "first_name": r[0],
            "last_name": r[1],
            "title": r[2],
            "company": r[3],
            "ls_company_website": r[4],
            "email": r[5],
            "lead_source": r[6],
            "street": r[7],
            "rating": r[8],
            "lead_owner": r[9],
        }
        
        message = dedent(f"""
        Outreach to the following prospect: 
                         
        Full Name: {row["first_name"]} {row["last_name"]}

        Email: {row["email"]}

        Job Title: {row["title"]}

        Company: {row["company"]}

        Company Website: {row["ls_company_website"]}

        Lead Owner: {row["lead_owner"]}
        """)

        task = client.tasks.trigger_task(
            agent_id=agent_id,
            message=message,
        )

        print("Triggered task: ", task.conversation_id)

        time.sleep(1)

print("Done!")

