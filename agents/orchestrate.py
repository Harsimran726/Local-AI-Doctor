from langgraph.graph import StateGraph, START, END 
from agents.child_agent import ChildDoctorAgent
from agents.medicine_agent import MedicineDoctorAgent   
from agents.receptionist_agent import ReceptionistAgent
from agents.research_agent import ResearchDoctorAgent
from agents.surgery_agent import SurgeryDoctorAgent
import os 
from dotenv import load_dotenv
load_dotenv()
import json 
from pydantic import error_wrappers


class OrchestrateAgent:
    def __init__(self):
        self.child_agent = ChildDoctorAgent()
        self.medicine_agent = MedicineDoctorAgent()
        self.surgery_agent = SurgeryDoctorAgent()
        self.reserach_agent = ResearchDoctorAgent()
        self.receptionist_agent = ReceptionistAgent()

    # X is the input from the the user, which is a string.
    def recepionist_node(self,x):
        try:
            response = self.receptionist_agent.invoke(x)
            return response
        except Exception as e:
            return None
        
    
    def child_node(self,x):
        try:
            response = self.child_agent.invoke(x)
            return response
        except Exception as e:
            return None
    
    def surgery_node(self,x):
        try:
            response = self.surgery_agent.invoke(x)
            return response
        except Exception as e:
            return None
        
    def medicine_node(self,x):
        try:
            response = self.medicine_agent.invoke(x)
            return response 
        except Exception as e:
            return None
        
    def reserch_node(self,x):
        try:
            response = self.reserach_agent.invoke(x)
            return response 
        except Exception as e:
            return None
    
    def universal_router_node(self,state:AgentState):
        try:
            next_agent = state.get("agent_call")
            if next_agent == "END":
                return "__end__"
            return next_agent

        except Exception as e:
            return "receptionist"
        
    

    def invoke(self,x):
        try:
            # first invoke goes to the receptionist agent, which will determine which agent to invoke next based on user relevant information . 
            workflow = StateGraph()

            router_map = {
                "receptionist": "receptionist",
                "child": "child",
                "surgery": "surgery",
                "medicine": "medicine",
                "research": "research",
                "__end__":"END"
            }

            workflow.add_node("receptionist", self.recepionist_node)
            workflow.add_node("child", self.child_node)
            workflow.add_node("surgery", self.surgery_node)
            workflow.add_node("medicine", self.medicine_node)
            workflow.add_node("research", self.reserch_node)

            workflow.add_edge(START,"receptionist")
            # add the conditional edges 
            nodes = ["receptionist","child","surgery","medicine","research"]
            for node in nodes:
                workflow.add_conditional_edge(
                    "START",
                    self.universal_router_node,
                router_map
                )
            workflow.add_edge("receptionist",END)


            # compile the workflow 
            app  = workflow.compile()

            return app 
            
        
        except Exception as e:
            print(f"ERROS {e}")
            return {'status':300,"message":f"ERROR {e}"}
        
