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


class GeneralDoctorAgent:
    def __init__(self):
        self.client = OpenAI()

    def invoke(self,x): 
        try:
            response = self.client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=x,
                reasoning_effort= "medium"
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
You are a Senior General Practitioner with more than 20+ Years of experience in the field of general medicine and have a deep understanding of various
medical conditions, treatments, Viruses, Bacteria and their impact on human health. You worked with the MIT and Harvard University for Medical Research 
, Vaccines Development for Viruses and Bacteria. You have published more than 100+ Reserach papers on various medical topics and have a strong background 
in clinical trials, epidemiology, and public health. 
</Persona>
<Context>
You have to work with the LOCAL DOCTOR agentic System, where you will be responsible for providing knowledge regarding the patient report and even give a advice, 
correct to the other doctor's agent's team. 
</Context>
<Task>
- You will get the Patient's report from the receptionist agent, 
- Provide them the Detailed Analysis of the report with the detailed reason behind the output with max 600 words. 
- You have access to the other agents where you can call them to get more information, and use thier thoughts to provide a better output.
- You also can get a report from the other agents and give them a advice regarding the report and provide a detailed reason behind the output.
- If you are not sure about the output then use the tools <TOOLS> to get more information and even ask to the other agents to get more information.
</Task>
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
"message_to_receptionist_agent": "Message to the receptionist agent to pass to the user with max 200 words. | None",
"message_to_other_agents": "Message to the other agents to get more information with max | None", # meessage to the agents 
"tool_call": "Instruction or data to pass to the tool | None". # just a tool call not any nested dictionary or json
"agent_call": "Instruction or data to pass to the other agent | None" # just a agent call not any nested dictionary or json , if two agents that keep them in a list even one agent keep them in a list
"message_to_tool": "Send a query to the tool to retrive the information from Vector DB | None"
}
</Output>


'''

def ask_to_general_agent():
    try:
        messages = [
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": "Patient intake summary (for doctor):\n- Name: not provided\n- Age: 21\n- Gender: Male\n- Chief complaint: Right-sided leg/hip pain\n- Location: Hip (patient believes)\n- Duration: More than 6 months (chronic)\n- Severity: Mild, stable (not worsening)\n- Pattern/relief: Feels more relief with hip hyperextension; sometimes reports a heavy feeling on the right side of the leg\n- Neurologic symptoms: No numbness, tingling, weakness, or swelling reported\n- Past medical history: None reported (no diabetes, HTN, thyroid disease)\n- Prior surgeries/injuries: None reported\n- Current medications/supplements: Magnesium; Vitamin D3 + K2\n- Lifestyle flags: None reported\n- Imaging/reports: None provided\n- OCR findings: none\nConversation log (patient messages):\n1) User: \"Hi\"\n2) Assistant: Greeted and asked how to assist\n3) User: \"i have a pain in leg right side\"\n4) Assistant: Asked age/gender\n5) User: \"21 year old man\"\n6) Assistant: Asked duration\n7) User: \"it more than 6 months\"\n8) Assistant: Asked if better/worse/stable\n9) User: \"it's staying the same mild , when i do the hyperextension than i get more relief\"\n10) Assistant: Asked exact location and neuro symptoms\n11) User: \"i'ts in hip i believe ,  i feel sometime heavy in right side of lef\"\n12) Assistant: Asked past medical conditions/prior injury\n13) User: \"no i don't have\"\n14) Assistant: Asked current medicines\n15) User: \"i'm only taking magnesium , vitamin d3 k2\"\nReason for routing: Adult with chronic localized hip pain without red flags — best routed to General Physician for musculoskeletal assessment; Medicine specialist included because patient provided current supplements/medication list for review.\nPlease review and advise next steps (exam, imaging needs, or medication review)"}
        ]
        # print(f"AGENT STSATE MESSAGE: {AgentState.agent_message}")
        general_agent = GeneralDoctorAgent()
        result = general_agent.invoke(messages)
        print(f"GENERAL AGENT RESULT: {result}")
        return result
    except Exception as e:
        print(f"Error in ask_to_general_agent: {e}")
        return None
  
ask_to_general_agent()