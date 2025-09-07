import os
import streamlit as st
import requests
import asyncio
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context # MODIFIED: Added Context import
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool

# --- Configuration and Secrets ---
# For local development, you can set these as environment variables.
# For Streamlit Community Cloud, set these in the secrets management.
# It's not secure to hardcode API keys directly in the script.
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", os.environ.get("OPENWEATHER_API_KEY"))

# --- Helper Function: Get Weather Data ---
def get_weather(city: str) -> dict:
    """
    Fetches the current weather forecast for a given city using the OpenWeatherMap API.
    
    Args:
        city: The name of the city.
        
    Returns:
        A dictionary containing the weather data, or an error message.
    """
    if not OPENWEATHER_API_KEY:
        return {"error": "OpenWeatherMap API key is not set."}
        
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'  # Use metric for Celsius
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return {"error": f"Could not fetch weather data for {city}. Please check the city name."}

# --- Caching the Language Model and Agent ---
# This prevents reloading the model and agent on every user interaction,
# which significantly speeds up the application.
@st.cache_resource
def load_agent():
    """
    Loads and configures the ReActAgent with the Groq LLM and the weather tool.
    
    Returns:
        An instance of ReActAgent.
    """
    if not GROQ_API_KEY:
        return None # Return None if the API key is not available

    llm = Groq(
        model="llama-3.1-8b-instant", # Updated to a more recent and capable model
        temperature=0.1 # A slight temperature for more natural responses
    )
    weather_tool = FunctionTool.from_defaults(fn=get_weather)
    
    agent = ReActAgent(
        tools=[weather_tool],
        llm=llm,
        verbose=True # Set to True to see the agent's thought process in the console
    )
    return agent

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Weather AI Chatbot",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Weather AI Chatbot")
st.markdown("Ask me anything about the current weather in any city!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you with the weather today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load the agent
agent = load_agent()

# Handle cases where API keys might be missing
if not GROQ_API_KEY or not OPENWEATHER_API_KEY:
    st.error("One or more API keys are missing. Please configure your secrets.")
    st.info("You need to set `GROQ_API_KEY` and `OPENWEATHER_API_KEY` in your Streamlit secrets.")
    st.stop() # Stop the app from running further if keys are missing

# Get user input
if prompt := st.chat_input("What is the weather like in London?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # MODIFIED: Created an async function to correctly call the agent
                async def get_response(query):
                    ctx = Context(agent) # Create the context required by the workflow agent
                    handler = agent.run(query, ctx=ctx) # Use the .run() method
                    response = await handler # Await the response handler
                    return response

                response = asyncio.run(get_response(prompt))
                response_text = str(response)
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I ran into a problem: {e}"})

