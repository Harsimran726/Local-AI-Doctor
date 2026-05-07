from langchain_core import tools
from langchain_core.memory import ConversationBufferMemory
import os 
from dotenv import load_dotenv
load_dotenv()
import re
import json 
import datetime 
from openai import OpenAI
from pydantic import error_wrappers


class SurgeryDoctorAgent:
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
        
