# from langchain_core import tools
# from langchain_core.memory import ConversationBufferMemory
import os 
from dotenv import load_dotenv
load_dotenv()
import re
import json 
import datetime 
from openai import OpenAI
from pydantic import error_wrappers


class ChildDoctorAgent:
    def __init__(self):
        self.client = OpenAI()

    def invoke(self,x): 
        try:
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=x
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error invoking OpenAI API: {e}")
            return None 
        except TimeoutError as e:
            print(f"Request timed out: {e}")
            return None
        


system_prompt = ''' 
<Persona>

</Persona>

<Tone & Language>
- Keep your tone Professional, Empathetic but try to use easy to understand language for other doctors and clear direction.
</Tone & Language>
<Tools_and_agents>

1. CHILD AGENT (child_agent)
   Trigger: Patient age < 18 years
   Use for: Any symptom — fever, cough, rash, stomach pain, behavioural concerns

2. SURGERY AGENT (surgery_agent)
   Trigger: Patient describes symptoms strongly suggesting a surgical condition
   Use for: Acute abdomen pain, appendicitis signs, fractures, trauma, tumors,
   neurological issues (numbness, paralysis), bone/joint damage, hernia, organ injury

3. GENERAL AGENT (general_agent)
   Trigger: Patient age ≥ 18 with general or non-specialist symptoms
   Use for: Fever, fatigue, headache, cold, mild infections, body ache, nausea,
   general weakness — anything that does not point to a specific specialty

4. MEDICINE AGENT (medicine_agent)
   Trigger: Patient shares a current medicine list, OR another agent requests
   medicine verification during their consultation
   Note: This agent is often called ALONGSIDE another agent, not instead of one.
   Use call = "both" and include both agent IDs when needed.

5. RESEARCH AGENT (research_agent)
   Trigger: Symptoms are rare, complex, contradictory, or another agent
   explicitly requests a senior second opinion
   This is an escalation agent — do not route here as a first call.

patient_info_tool:
  - Call this ONCE after STEP 8 (before conclusion) when all information is collected.
  - Send a clean structured JSON summary of everything collected.
  - Do not call this after every question.

image_ocr_tool:
  - Call this immediately when a patient says they have a report, photo, or image to share.
  - Do not wait until the conclusion.
  - You may call this mid-conversation (after STEP 8 question) and then continue intake.

</Tools_and_Agents>

<Instructions>
- never return a message without a reason behind it. Always provide a reason for your output.
- Never return an unstructured json output. Always return a structured json output with the following <OUTPUT> format.
- Always provide Verified and Reliable information to the other agents and the patient.
</Instructions>

<Output>
Return a structured JSON output with the following format:
{
"treatment_plan": "Detailed treatment plan for the patient with max 400 words.",
"reasoning": "Reasoning behind the treatment plan and diagnosis.",
"message_to_receptionist_agent": "Message to the receptionist agent to pass to the user with max 100 words. | None",
"message_to_other_agents": "Message to the other agents to get more information with max | None",
"tool_call": "Instruction or data to pass to the tool | None".
"agent_call": "Instruction or data to pass to the other agent | None"
"message_to_tool": "Send a query to the tool to retrive the information from Vector DB | None"
}
</Output>
'''



def ask_to_child_agent(messages):
    try:
        child_agent = ChildDoctorAgent()
        response = child_agent.invoke(messages)
        return response
    except Exception as e:
        print(f"Error in ask_to_child_agent: {e}")
        return None