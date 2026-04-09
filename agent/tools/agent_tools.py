import os
from utils.logger_handler import logger

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
import requests

user_ids = [str(i) for i in range(1001, 1011)]
month_arr = [
    "2025-01",
    "2025-02",
    "2025-03",
    "2025-04",
    "2025-05",
    "2025-06",
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
]
external_data = {}


@tool(
    description="This tool is used to summarize the query with RAG method. "
    "The input should be a question or a query, and the output will be the "
    "summarized answer based on the knowledge in the vector store."
)
def rag_summarize(query: str) -> str:
    rag_service = RagSummarizeService()
    return rag_service.rag_summarize(query)


@tool(
    description=(
        "This tool is used to get the weather information of a city. "
        "The input should be the name of the city, and the output will be the "
        "weather information of that city."
    )
)
def get_weather(city: str, lat: str, lon: str) -> str:
    api_key = agent_conf.get("openweather_api_key", None)
    if not api_key or api_key.startswith("${"):
        logger.error("OpenWeather API key is not configured.")
        return "Weather information is currently unavailable."
    logger.info(f"Get the api key for openweather: {api_key is not None}")
    resp = requests.get(
        f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=hourly,daily&appid={api_key}"
    )
    if resp.status_code != 200:
        logger.error(
            f"OpenWeather API request failed with status {resp.status_code}: {resp.text}"
        )
        return f"Unable to fetch weather data for {city}. Please try again later."

    data = resp.json()
    weather_desc = data["current"]["weather"][0]["description"]
    temp = data["current"]["temp"] - 273.15  # Convert from Kelvin to Celsius
    humidity = data["current"]["humidity"]
    return f"The current weather in {city} is {weather_desc} with a temperature of {temp}°C and humidity of {humidity}%."


@tool(
    description="This tool is used to get the name of the user's city."
    "The input should be the user's IP address,"
    "and the output will be the 'city', 'lat' and 'lon' corresponding to that IP address.",
)
def get_user_location(ip: str) -> str:
    resp = requests.get(f"http://ip-api.com/json/{ip}")
    data = resp.json()
    res = f"city: {data.get('city', 'unknown')}, lat: {data.get('lat', 'unknown')}, lon: {data.get('lon', 'unknown')}"
    return res


@tool(description="Obtain the user id, return in string format")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(
    description="Obtain a random user id from a predefined list of user ids, return in string format"
)
def get_random_user_id() -> str:
    return random.choice(user_ids)


@tool(description="Obtain the current month, return in string format")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(
                f"External data file not found at {external_data_path}"
            )

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id = arr[0].replace('"', "")
                feature = arr[1].replace('"', "")
                efficiency = arr[2].replace('"', "")
                consumables = arr[3].replace('"', "")
                comparison = arr[4].replace('"', "")
                time = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "feature": feature,
                    "efficiency": efficiency,
                    "consumables": consumables,
                    "comparison": comparison,
                }


@tool(
    description=(
        "This tool is used to fetch the external data based on the user id and month. "
        "The input should be a user id and a month, and the output will be the "
        "corresponding external data if available."
    )
)
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"Data not found for user_id: {user_id}, month: {month}")
        return ""


@tool(
    description=(
        "No input, no output. This is a placeholder tool for filling the context for report generation."
    )
)
def fill_context_for_report():
    return "fill_context_for_report is called"
