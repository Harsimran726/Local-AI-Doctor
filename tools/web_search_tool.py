from firecrawl import Firecrawl
import firecrawl
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()
import os 
from pydantic import BaseModel

class Websearch(BaseModel):
    query: str 
    url: list[str] 
    title: list[str]
    description: list[str]


class WebSearchTool:
    def __init__(self):
        self.name = "Web Search Tool"
        self.description = "A tool for performing web searches using the Firecrawl API."

    def connection_firecrawl(self):
        try:
            firecrawl = Firecrawl(api_key=os.getenv("firecrawl_api_key"))
            return firecrawl
        except Exception as e:
            print(f"Error connecting to Firecrawl: {e}")
            return "Connection failed"

    def search_firecrawl(self, query:str):
        try:
            print(f"Performing search for query: {query}")
            firecrawl = self.connection_firecrawl()
            results = firecrawl.search(
            query=query,
            limit=4, )
            web_results = [search for search in results.web]
            return web_results
        # output comes in this way :-
        # [SearchResultWeb(url='https://en.wikipedia.org/wiki/Paris', title='Paris - Wikipedia', description="As the capital of France, Paris is the seat of France's national government ; Both houses of the French Parliament ; France's highest courts are located in Paris.", category=None), 
        # SearchResultWeb(url='https://www.youtube.com/shorts/doFTuSWo7rE', title='What is the capital of FRANCE? - YouTube', description="What is the capital of FRANCE? 105K. Dislike. 4,230. Share. Video unavailable. This content isn't available. Skip video.", category=None), 
        # SearchResultWeb(url='https://www.coe.int/en/web/interculturalcities/paris', title='Paris, France - Intercultural Cities Programme - The Council of Europe', description='Paris is the capital and most populous city of France. Situated on the Seine River, in the north of the country, it is in the centre of the Île-de-France ...', category=None)]
        except Exception as e: 
            return {"message": f"An error occurred while performing the search: {e}"} 
        except TimeoutError as e:
            print(f"Request timed out: {e}")
            return {"message":f"Search request timeed out, please try again later. {e}"}
        
