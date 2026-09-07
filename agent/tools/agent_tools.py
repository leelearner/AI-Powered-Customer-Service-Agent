import os
from utils.logger_handler import logger

from langchain_core.tools import tool

from rag.rag_service import get_rag_service
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HTTP_TIMEOUT = agent_conf.get("http_timeout_seconds", 5)


def _build_http_session() -> requests.Session:
    """Session that retries transient failures before giving up.

    Retry belongs here rather than on the graph node: the node-level degradation
    decorator catches exceptions inside the node, so LangGraph's retry_policy —
    which wraps from the outside — would never see them.
    """
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


http_session = _build_http_session()

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
    return get_rag_service().rag_summarize(query)


@tool(
    description=(
        "This tool is used to get the weather information of a city. "
        "The input should be the name of the city, and the output will be the "
        "weather information of that city."
    )
)
def get_weather_tool(city: str, lat: str, lon: str) -> str:
    api_key = agent_conf.get("openweather_api_key", None)
    if not api_key or api_key.startswith("${"):
        logger.error("OpenWeather API key is not configured.")
        return "Weather information is currently unavailable."
    logger.info(f"Get the api key for openweather: {api_key is not None}")
    try:
        resp = http_session.get(
            f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=hourly,daily&appid={api_key}",
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException as e:
        logger.error(f"OpenWeather API request failed: {e}")
        return f"Unable to fetch weather data for {city}. Please try again later."

    if resp.status_code != 200:
        logger.error(
            f"OpenWeather API request failed with status {resp.status_code}: {resp.text}"
        )
        return f"Unable to fetch weather data for {city}. Please try again later."

    try:
        data = resp.json()
        weather_desc = data["current"]["weather"][0]["description"]
        temp = data["current"]["temp"] - 273.15  # Convert from Kelvin to Celsius
        humidity = data["current"]["humidity"]
    except (ValueError, KeyError, IndexError) as e:
        logger.error(f"Unexpected OpenWeather response shape: {e}")
        return f"Unable to fetch weather data for {city}. Please try again later."

    return f"The current weather in {city} is {weather_desc} with a temperature of {temp}°C and humidity of {humidity}%."


@tool(
    description="This tool is used to get the name of the user's city."
    "The input should be the user's IP address,"
    "and the output will be the 'city', 'lat' and 'lon' corresponding to that IP address.",
)
def get_user_location(ip: str) -> dict:
    unknown = {"city": "unknown", "lat": "", "lon": ""}
    try:
        resp = http_session.get(f"http://ip-api.com/json/{ip}", timeout=HTTP_TIMEOUT)
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.error(f"IP geolocation lookup failed for {ip}: {e}")
        return unknown

    return {
        "city": data.get("city", "unknown"),
        "lat": data.get("lat", ""),
        "lon": data.get("lon", ""),
    }


# DEPRECATED: the graph no longer calls these three. They returned a random user
# and a random month, so "my June report" could return another user's March data.
# `get_user_id` now reads the session's selected user and `get_month` uses the month
# extracted from the query. Kept for reference, as with react_agent.py / middleware.py.
@tool(description="Obtain the user id, return in string format")
def get_user_id_tool() -> str:
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


def available_user_ids() -> list[str]:
    """User ids actually present in the external data, sorted."""
    generate_external_data()
    return sorted(external_data.keys())


def available_months(user_id: str = "") -> list[str]:
    """Months present in the external data, sorted ascending.

    Scoped to one user when given, otherwise the union across all users.
    """
    generate_external_data()
    if user_id and user_id in external_data:
        return sorted(external_data[user_id].keys())
    months: set[str] = set()
    for record in external_data.values():
        months.update(record.keys())
    return sorted(months)


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
