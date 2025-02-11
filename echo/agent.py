from crewai import Crew
from pydantic import Field
from echo.settings import MAX_RETRIES


class EchoAgent(Crew):
    max_retries: int = Field(MAX_RETRIES, description="Maximum number of retries")
    
    async def kickoff_async(self, inputs):
        retries = self.max_retries
        while retries > 0:
            try:
                response = await super().kickoff_async(inputs)  # Use `super()` to call parent method
                return response
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise e
                print(f"Retrying due to error: {e}")
        
    def kickoff(self, inputs):
        retries = self.max_retries
        while retries > 0:
            try:
                response = super().kickoff(inputs)  # Use `super()` to call parent method
                return response
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise e
                print(f"Retrying due to error: {e}")
