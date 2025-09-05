import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
load_dotenv(find_dotenv())

model = ChatAnthropic(model="claude-sonnet-4-20250514",
                      temperature= 0.5,
                      max_tokens=2000,
                      max_retries=2)

print("Enter your travel itinerary (press Enter twice when done):")
itinery_lines = []
while True:
    line = input()
    if line == "":
        break
    itinery_lines.append(line)
itinery_day1 = "\n".join(itinery_lines)

itinery_template ="""
From the following itinery, extract the following information
1. Leave_time : when to leave , if there is an actual time written, use it, if not, write unknown
2. leave_from : where to leave from, if there is an actual place written, use it, if not, write unknown
3. places_to_visit : list of places to visit. If there are more than one, put them in square brackets like ["city centre","church"]
4. restaurants : places to eat at, if there is an actual place written, use it, if not, write Any. If there are more than one, put them in square brackets like ["Cafe Paash","Montecito Breeze"]

Format the output as JSON with following keys
leave_time
leave_from
places_to_visit
restaurants

itinery = {itinery}
"""

prompt = ChatPromptTemplate.from_template(itinery_template)
messages = prompt.format_messages(itinery=itinery_day1)

response = model.invoke(messages)
#print(response.content)


################  Using langchain parsers ######################
leave_time_schema = ResponseSchema(name="leave_time",
                                   description="when to leave , if there is an actual time written, use it, if not, write unknown")

leave_from_schema = ResponseSchema(name="leave_from",
                                   description="where to leave from, if there is an actual place written, use it, if not, write unknown")
places_to_visit_schema = ResponseSchema(name="places_to_visit",
                                   description="list of places to visit. If there are more than one, put them in square brackets like ['city centre','church']")
restaurants_schema = ResponseSchema(name="restaurants",
                                   description="places to eat at, if there is an actual place written, use it, if not, write Any. If there are more than one, put them in square brackets like ['Cafe Paash','Montecito Breeze']")

response_schemas = [leave_time_schema, leave_from_schema, places_to_visit_schema, restaurants_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions
#print(format_instructions)

itinery_template_revised = """
From the following itinery, extract the following information
1. Leave_time : when to leave , if there is an actual time written, use it, if not, write unknown
2. leave_from : where to leave from, if there is an actual place written, use it, if not, write unknown
3. places_to_visit : list of places to visit. If there are more than one, put them in square brackets like ["city centre","church"]
4. restaurants : places to eat at, if there is an actual place written, use it, if not, write Any. If there are more than one, put them in square brackets like ["Cafe Paash","Montecito Breeze"]

Format the output as JSON with following keys:
leave_time
leave_from
places_to_visit
restaurants

itinery = {itinery}
{format_instructions}
"""
#print(format_instructions)
updated_prompt = ChatPromptTemplate.from_template(itinery_template_revised)
new_messages = updated_prompt.format_messages(itinery=itinery_day1, format_instructions=format_instructions)

response = model.invoke(new_messages)
#print(response.content)
#print(type(response.content))
output_dict = output_parser.parse(response.content)

print("\n" + "="*40)
print("EXTRACTED ITINERARY INFORMATION")
print("="*40)
print(f"Leave Time: {output_dict['leave_time']}")
print(f"Leave From: {output_dict['leave_from']}")
print(f"Places to Visit: {', '.join(output_dict['places_to_visit']) if isinstance(output_dict['places_to_visit'], list) else output_dict['places_to_visit']}")
print(f"Restaurants: {', '.join(output_dict['restaurants']) if isinstance(output_dict['restaurants'], list) else output_dict['restaurants']}")
print("="*40)
