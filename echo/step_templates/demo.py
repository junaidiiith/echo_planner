import copy
from tqdm.asyncio import tqdm
from crewai import Crew, LLM
from crewai.crews.crew_output import CrewOutput
from echo.constants import *
from pydantic import BaseModel, Field
from typing import Dict, List
from echo.utils import add_pydanctic_structure, format_response
from echo.step_templates.generic import Transcript, save_client_call_data, embed_client_call_data
from echo.utils import get_crew as get_crew_obj
import echo.utils as utils

# ------------------------------------------------------------------------------------------------ #
## Seller Data
# ------------------------------------------------------------------------------------------------ #

class Feature(BaseModel):
	name: str = Field(..., title="Feature Name", description="The name of the feature of the product or service.")
	description: str = Field(..., title="Feature Description", description="A description of the feature.")

class FeatureWithPainPoint(BaseModel):
    name: str = Field(..., title="Feature Name", description="The name of the feature of the product or service.")
    description: str = Field(..., title="Feature Description", description="A description of the feature.")
    pain_point: str = Field(..., title="Pain Point", description="The pain point or challenge that the feature addresses.")

class FeaturesAnticipated(BaseModel):
	features: List[Feature] = Field(..., title="Features", description="A list of features of the product or service that will be demonstrated during the sales demo.")

class SkepticalFeature(Feature):
    name: str = Field(..., title="Feature Name", description="The name of the feature of the product or service.")
    description: str = Field(..., title="Feature Description", description="A description of the feature.")
    objection: str = Field(..., title="Objection", description="The objection or concern raised by the buyer.")
 
class ObjectionResolution(BaseModel):
    objection: str = Field(..., title="Objection", description="The objection raised by the buyer.")
    resolution: str = Field(..., title="Resolution", description="The resolution provided by the seller.")
    status: bool = Field(..., title="Status", description="The status of the resolution, True if resolved, False if not resolved.")

class SellerDataExtracted(BaseModel):
	demo_features: List[Feature] = Field(..., title="Features", description="A list of features of the product or service that will be demonstrated during the sales demo.")
	objection_handling: List[ObjectionResolution] = Field(..., title="Objections", description="The objections raised by the buyer and their status.")
	success_stories: List[str] = Field(..., title="Success Stories", description="A list of success stories or case studies related to the product or service.")
	assets_used: List[str] = Field(..., title="Assets Used", description="A list of assets used during the sales demo, such as presentations, videos, or documents.")
	customizations: List[str] = Field(..., title="Customizations", description="A list of customizations or personalized features offered to the buyer.")
	engagement_questions: List[str] = Field(..., title="Engagement Questions", description="A list of questions to engage the buyer during the sales demo.")

# ------------------------------------------------------------------------------------------------ #
## Buyer Data
# ------------------------------------------------------------------------------------------------ #


class StakeholderEngagement(BaseModel):
	stakeholder : str = Field(..., title="Stakeholder", description="The stakeholder's name.")
	engagement_level: str = Field(..., title="Engagement Level", description="The engagement level of the stakeholder during the sales demo - high, medium, or low.")


class BuyerDataExtracted(BaseModel):
	interested_features: List[FeatureWithPainPoint] = Field(..., title="Interested Features", description="The features of the product or service that stakeholders from the buyer are interested in.")
	skeptical_features: List[SkepticalFeature] = Field(..., title="Skeptical Features", description="The features of the product or service that stakeholders from the buyer are skeptical or have objections, concerns about.")
	objections_raised: List[str] = Field(..., title="Objections", description="A list of objections raised by the buyer.")
	stakeholder_engagement: List[StakeholderEngagement] = Field(..., title="Stakeholders' Engagement", description="The engagement level of the stakeholders during the sales demo.")
	feedback: str = Field(..., title="Feedback", description="The feedback provided by the buyer after the sales demo.")	
 

agent_templates = {
	RESEARCH: {
		"PreparationAgent": dict(
			role="Sales Demo Calls Expert",
			goal="The goal of this agent is to prepare for the sales demo calls by mapping the features of the product or service to the goals, requirements, questions, objections, pain points, challenges, and return on investment.",
			backstory=(
				"You are an expert in preparing for sales demo calls. "
				"You have an in-depth understanding of how to sell the features of a product and service to potential customers."
				"You have the ability to sell the benefits of the product or service according to the requirements, goals, pain points of the customers such that they are satisfied and convinced to make a purchase."
				"Using these abilities, you will map the features of the product or service to the goals, requirements, questions, objections, pain points, challenges, and return on investment."
			)
		),
	},
	SIMULATION: {
		"CallSimulationAgent": dict(
			role="Sales Call Simulation Specialist",
			goal="Simulate a very elaborated, detailed call between buyer and {seller}.",
			backstory=(
				"You are an expert in simulating realistic sales calls."
				"You have been tasked with simulating a detailed call between buyer and {seller}."
				"Your goal is to provide a realistic and engaging simulation of the call."
				"The sales call simulation should be tailored to address the buyer's specific goals, requirements, pain points, and challenges."
			)
		)
	},
	ANALYSIS: {
		"CallAnalysisAgent": dict(
			role="Sales Call Analysis Specialist",
			goal="Analyze the sales call between buyer and {seller} to identify the key features, pain points, challenges, objections, insights and areas of improvement.",
			backstory=(
				"You are an expert in analyzing demo sales calls and identifying key insights."
				"Your goal is to analyze the sales call between buyer and {seller}."
				"Your goal is identify the key pain points, challenges, objections, key insights, areas of improvement, and potential strategies for future calls."
			)
		)
	}
}

task_templates = {
	RESEARCH: {
		"FeatureExtraction": dict(
			name="Extract the features of the {seller}'s product",
			description=(
				"Extract the features of the {seller}'s product or service that will be demonstrated during the sales demo for the {buyer}\n"
				"You are supposed to extract the features from the information provided below.\n"
				"You are provided with the following context -\n"
				"Transcript of the discovery call between {buyer} and {seller}\n"
				"{seller}'s research information {seller_research}\n"
				"{seller}'s pricing model {seller_pricing}\n"
				"\n{buyer}'s research information\n{buyer_research}\n"
				"\nCompetitive Information:\n{competitive_info}\n"
				"\n{buyer}'s Data from the analysis of previous discovery call:\n{discovery_analysis_buyer_data}\n"
    			"\n{seller}'s Data from the analysis of previous discovery call:\n{discovery_analysis_seller_data}\n"
			),
			expected_output=(
				"A list of features of the {seller}'s product or service that will be demonstrated during the sales demo.\n"
				"The response should conform to the provided schema.\n"
				"You need to extract the following information in the following pydantic structure -\n"
				"{pydantic_structure}\n"
				"Make sure there are no comments in the response JSON and it should be a valid JSON."
			),
			output_pydantic=FeaturesAnticipated,
			agent="PreparationAgent"
		)
	},
	SIMULATION: {
		"CallSimulationTask": dict(
			name="Simulate demo Call",
			description=(
				"Simulate a very elaborated, detailed demo call between {buyer} and {seller}. "
				"Assume multiple stakeholder involved from the {buyer}'s side. "
				"You are provided with the {buyer}'s and {seller}'s information as well as the discovery call information."
    
				"The information from the discovery call will be used to simulate the demo call."
				"\n{buyer}'s Data from the analysis of previous discovery call:\n{discovery_analysis_buyer_data}\n"
    			"\n{seller}'s Data from the analysis of previous discovery call:\n{discovery_analysis_seller_data}\n"
				"You are also provided with potential features that can address pain points of the buyer below - \n{demo_features}\n"
    
				
				"In the call, {seller}'s sales person will aim to provide the features of the product or service that will be demonstrated during the sales demo."
				"The {seller}'s team person is supposed to be very confident and knowledgeable about the product or service during the demo."
				"In order to simulate the demo call, you need to follow the following steps - "
				"1. Think about what is the objective of a general demo sales call after the demo call."
				"2. Use the context, i.e., {seller}'s features and buyer's requirements, pain points and their mappings to simulate the call that fulfills the objectives of a demo call.\n"
    
				"The simulation MUST cover the following aspects -\n"
				"1. The {seller}'s product features that MUST be clearly demonstrated during the sales demo.\n"
    			"2. The {buyer} raises some objections during the sales demo and they MUST be handled by the {seller}.\n"
				"3. The {seller} MUST mention some of their relevant success stories or case studies related to the product or service.\n"
				"4. The {seller} should clarify the used during the demo assets like presentations, videos, or documents during the sales demo.\n"
				"5. The {seller} MUST offer customizations or personalized features to the buyer based on {buyer}'s objections and pain points which the existing features do not cover.\n"
				"6. The {seller} MUST engage the buyer with relevant questions. "
				

				"Your goal is to provide a realistic and engaging simulation of a demo call."
				"The sales call MUST clearly cover the {buyer}'s goals, pain points and objections. "
				"The contents of each message should be as detailed and realistic as possible like a human conversation. "
				"The call should be structured and flow naturally like a real demo call. "
				"The call should have a smooth flow and should be engaging and informative. "
				"The call should not end abruptly and should have a proper conclusion with next steps according to how the call went. "
		
				"You are provided with the following context -\n"
				"{seller}'s product features: {demo_features}\n"
				
			),
			expected_output=(
				"A realistic simulation of the demo call between {buyer} and {seller}."
				"You need to extract the following information in the following pydantic structure -\n"
				"{pydantic_structure}\n"
				"Make sure there are no comments in the response JSON and it should be a valid JSON."
			),
			agent="CallSimulationAgent",
			output_pydantic=Transcript
		)
	},
	ANALYSIS: {
		"BuyerAnalysis": dict(
			name="Analyze the Demo Call and extract Buyer data",
			description=(
				"Analyze the sales call between {buyer} and {seller} to extract the buyer's data.\n"
				"You are provided with the transcript of the demo call between {buyer} and {seller}.\n"
				"Your goal is to analyze the call and provide insights to the sales team.\n"
				
    			"The analysis should include -\n"
				"1. The features of the product or service that stakeholders from the {buyer} are interested in. \n"
				"2. The features of the product or service that stakeholders from the {buyer} are skeptical or have objections, concerns about. \n"
    			"3. The objections raised by the buyer \n"
       			"4. Engagement level of all the different stakeholder\n"
          		"5. Feedback provided by the buyer.\n"
				
    			"You are provided with the following context -\n"
				"{demo_transcript}\n"
			),
			expected_output=(
				"An analysis of the sales call between {buyer} and {seller}.\n"
				"The response should conform to the provided schema.\n"
				"You need to extract the following information in the following pydantic structure -\n"
				"{pydantic_structure}\n"
				"Make sure there are no comments in the response JSON and it should be a valid JSON."
			),
			agent="CallAnalysisAgent",
			output_pydantic=BuyerDataExtracted
		),
		"SellerAnalysis": dict(
			name="Analyze the Demo Call and extract Seller data",
			description=(
				"Analyze the sales call between {buyer} and {seller} to extract the seller's data.\n"
				"You are provided with the transcript of the demo call between {buyer} and {seller}.\n"
				"Your goal is to analyze the call and provide insights to the sales team.\n"
				"The analysis should include -\n"
		
				"1. All the features of the product or service that were demonstrated during the sales demo.\n"
				"2. The objections raised by the buyer and how they were handled.\n"
				"3. Success stories or case studies related to the product or service.\n"
				"4. Assets used during the sales demo.\n"
				"5. Customizations or personalized features offered to the buyer.\n"
				"6. Engagement questions to engage the buyer during the sales demo.\n"
	
				
				"You are provided with the following context -\n"
				"{demo_transcript}\n"
			),
			expected_output=(
				"An analysis of the sales call between {buyer} and {seller}.\n"
				"The response should conform to the provided schema.\n"
				"You need to extract the following information in the following pydantic structure -\n"
				"{pydantic_structure}\n"
				"Make sure there are no comments in the response JSON and it should be a valid JSON."
			),
			context=["BuyerAnalysis"],
			agent="CallAnalysisAgent",
			output_pydantic=SellerDataExtracted
		)
	}
}


buyer_call_format = (
    "Interested Features: {demo_analysis_buyer_data.interested_features}\n"
    "Objections Raised during call: {demo_analysis_buyer_data.objections_raised}\n"
    "Skeptical Features: {demo_analysis_buyer_data.skeptical_features}\n"
    "Stakeholder Engagement: {demo_analysis_buyer_data.stakeholder_engagement}\n"
    "Feedback: {demo_analysis_buyer_data.feedback}\n"
)

buyer_keys = {
	"Interested Features": "demo_analysis_buyer_data.interested_features",
	"Objections Raised during call": "demo_analysis_buyer_data.objections_raised",
	"Skeptical Features": "demo_analysis_buyer_data.skeptical_features",
	"Stakeholder Engagement": "demo_analysis_buyer_data.stakeholder_engagement",
	"Feedback": "demo_analysis_buyer_data.feedback"
}


seller_call_format = (
    "Demo Features: {demo_analysis_seller_data.demo_features}\n"
	"Objection Handling: {demo_analysis_seller_data.objection_handling}\n"
	"Success Stories: {demo_analysis_seller_data.success_stories}\n"
	"Assets Used: {demo_analysis_seller_data.assets_used}\n"
	"Customizations: {demo_analysis_seller_data.customizations}\n"
	"Engagement Questions: {demo_analysis_seller_data.engagement_questions}\n"
)

seller_keys = {
	"Demo Features": "demo_analysis_seller_data.demo_features",
	"Objection Handling": "demo_analysis_seller_data.objection_handling",
	"Success Stories": "demo_analysis_seller_data.success_stories",
	"Assets Used": "demo_analysis_seller_data.assets_used",
	"Customizations": "demo_analysis_seller_data.customizations",
	"Engagement Questions": "demo_analysis_seller_data.engagement_questions"
}


def get_client_data_to_embed(client_name, user_type):
    data = utils.get_client_data(client_name)
    if user_type == BUYER:
        return utils.replace_keys_with_values(buyer_call_format, data)
    else:
        return utils.replace_keys_with_values(seller_call_format, data)


def get_client_data_to_save(client_name, user_type):
    data = utils.get_client_data(client_name)
    if user_type == BUYER:
        return utils.get_nested_key_values(buyer_keys, data)
    else:
        return utils.get_nested_key_values(seller_keys, data)



def get_crew(step: str, llm: LLM, **crew_config) -> Crew:
	assert step in [RESEARCH, SIMULATION, EXTRACTION, ANALYSIS], \
		f"Invalid step type: {step} Must be one of 'research', 'simulation', 'extraction', 'analysis'"
	
	return get_crew_obj(
		agent_templates=agent_templates[step],
		task_templates=task_templates[step],
		llm=llm,
		**crew_config
	)


def process_research_data_output(response: CrewOutput):
	features = format_response(response.tasks_output[0])
	return {
		"demo_features": features
 	}


def process_analysis_data_output(response: CrewOutput):
	data = {
		"demo_analysis_buyer_data": format_response(response.tasks_output[0]),
		"demo_analysis_seller_data": format_response(response.tasks_output[1])
    }
	return data


async def aget_research_data_for_client(inputs: dict, llm: LLM, **crew_config):
	if utils.check_data_exists(inputs['buyer']):
		data: Dict = utils.get_client_data(inputs['buyer'])
		if 'demo_features' in data:
			return data
		else:
			inputs.update(copy.deepcopy(data))
	else:
		raise ValueError(f"Data for client {inputs['buyer']} does not exist.")
	
	crew = get_crew(RESEARCH, llm, **crew_config)
	add_pydanctic_structure(crew, inputs)
	response = await crew.kickoff_async(inputs=inputs)
	data.update(process_research_data_output(response))
	return data


async def aget_simulation_data_for_client(inputs: dict, llm: LLM, **crew_config):
	assert all([k in inputs for k in [
		"seller", 
		"buyer", 
		"seller_research", 
		"seller_pricing", 
		"buyer_research", 
		"competitive_info", 
		"anticipated_qopcs", 
	]]), "Invalid input data for simulation"
	if utils.check_data_exists(inputs['buyer']):
		data = utils.get_client_data(inputs['buyer'])
		if 'demo_transcript' in data:
			return data
	
	crew = get_crew(SIMULATION, llm, **crew_config)
	add_pydanctic_structure(crew, inputs)
	response = await crew.kickoff_async(inputs=inputs)
	return {
		"demo_transcript": format_response(response.tasks_output[0])
    }


async def aanalyze_data_for_client(inputs: dict, llm: LLM, **crew_config):
	assert all([k in inputs for k in [
		"seller", 
		"buyer", 
		"seller_research", 
		"seller_pricing", 
		"buyer_research", 
		"competitive_info", 
		"demo_transcript"
	]]), "Invalid input data for simulation"
	if utils.check_data_exists(inputs['buyer']):
		data = utils.get_client_data(inputs['buyer'])
		if 'demo_analysis_buyer_data' in data:
			return data
	
	crew = get_crew(ANALYSIS, llm, **crew_config)
	add_pydanctic_structure(crew, inputs)
	response = await crew.kickoff_async(inputs=inputs)
	return process_analysis_data_output(response)


async def aget_research_data(
	clients: List[str],
	inputs: dict,
	llm: LLM,
	**crew_config
):
	research_data: Dict[str, Dict] = dict()
	for client in tqdm(clients, desc="Get Research Data"):
		data = copy.deepcopy(inputs)
		data['buyer'] = client
		response = await aget_research_data_for_client(
			data,
			llm,
			**crew_config
		)
		data.update(response)
		utils.save_client_data(client, data)
		research_data[client] = data
  	
	
	return research_data


async def asimulate_data(clients: List[str], inputs: dict, llm: LLM, **crew_config):
	research_data = await aget_research_data(clients, inputs, llm, **crew_config)
	simulation_data: Dict[str, Dict] = dict()
	for client in tqdm(clients, desc="Simulating Data"):
		print(f"Simulating Data for {client}")
		data = copy.deepcopy(research_data[client])
		data.update(inputs)
		data['buyer'] = client
		response = await aget_simulation_data_for_client(data, llm, **crew_config)
		data.update(response)
		utils.save_client_data(client, data)
		simulation_data[client] = data

	return simulation_data



async def aget_analysis(
	clients: List[str], 
	inputs: dict, 
	llm: LLM, 
	**crew_config
):
	analysis_data = dict()
	simulated_data = await asimulate_data(clients, inputs, llm, **crew_config)
	
	for client in tqdm(clients, desc="Analyzing Data"):
		data: Dict = copy.deepcopy(simulated_data[client])
		data.update(inputs)
		data['buyer'] = client
		response = await aanalyze_data_for_client(data, llm, **crew_config)
		data.update(response)
		utils.save_client_data(client, data)
		analysis_data[client] = data
  
		save_client_data_to_db(client, inputs['seller'])
	
	return analysis_data


def save_client_data_to_db(client_name, seller_name):
    inputs = {
        SELLER: seller_name,
        BUYER: client_name,
    }
    save_client_call_data(client_name, BUYER, DEMO, inputs, get_client_data_to_save(client_name, BUYER))
    save_client_call_data(client_name, SELLER, DEMO, inputs, get_client_data_to_save(client_name, SELLER))
    embed_client_call_data(client_name, BUYER, DEMO, inputs, get_client_data_to_embed(client_name, BUYER))