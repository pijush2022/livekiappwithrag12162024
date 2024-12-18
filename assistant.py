import logging
import pickle
import os
import numpy as np
import aiohttp
import asyncio

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, rag, silero, cartesia

# Set up logging
logger = logging.getLogger("rag-assistant")
logging.basicConfig(level=logging.INFO)

# Load the RAG index and data
try:
    annoy_index = rag.annoy.AnnoyIndex.load("vdb_data")  # See build_data.py
    logger.info("Annoy index loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Annoy index: {e}")
    raise e

embeddings_dimension = 1536

try:
    with open("my_data.pkl", "rb") as f:
        paragraphs_by_uuid = pickle.load(f)
    logger.info("Paragraphs data loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load paragraphs data: {e}")
    raise e


async def _create_embeddings(input: str) -> np.ndarray:
    openai_api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Hardcode your API key for testing

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": input,
        "model": "text-embedding-ada-002",  # Using a fallback model
    }

    try:
        logger.info(f"Creating embedding for input: {input[:100]}...")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response_json = await response.json()
                logger.info(f"Full API response: {response_json}")

                # Check if 'error' key is present in the response
                if "error" in response_json:
                    logger.error(f"API Error: {response_json['error']['message']}")
                    return None

                # Check if 'data' key is present in the response
                if "data" in response_json and response_json["data"]:
                    return response_json["data"][0]["embedding"]

                # Unexpected response structure
                logger.error(f"Unexpected response structure: {response_json}")
                return None

    except Exception as e:
        logger.error(f"Exception while creating embeddings: {e}")
        return None




async def entrypoint(ctx: JobContext):
    """Entrypoint for the voice assistant."""
    async def _enrich_with_rag(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        user_msg = chat_ctx.messages[-1]
        user_embedding = await _create_embeddings(user_msg.content)

        if user_embedding is not None:
            try:
                result = annoy_index.query(user_embedding, n=1)[0]
                paragraph = paragraphs_by_uuid.get(result.userdata, None)

                if paragraph:
                    logger.info(f"Enriching with RAG: {paragraph}")
                    rag_msg = llm.ChatMessage.create(
                        text="Context:\n" + paragraph,
                        role="assistant",
                    )
                    chat_ctx.messages[-1] = rag_msg
                    chat_ctx.messages.append(user_msg)
                else:
                    logger.error("No relevant paragraph found in the vector database.")
            except Exception as e:
                logger.error(f"Failed to query the vector database: {e}")
        else:
            logger.error("Embedding creation failed, skipping RAG enrichment.")

    # Initial chat context
    initial_ctx = llm.ChatContext().append(
        role="system",
        # text=(
        #     "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
        #     "You should use short and concise responses, avoiding usage of unpronounceable punctuation. "
        #     "Use the provided context to answer the user's question if needed."
        # ),
         text=(
            '''You’re Annie, a country music expert, guiding folks around Birmingham, Alabama. Today is December 7th 2024.
            Use the context where possible to answer the question. do not use words such as y'all or those that are hard to pronounce.
            Answer in a simple yes or no, if you have to elaborate make it succinct and keep it under 4 lines. Provide complete, accurate, and relevant information, ensuring no key details are missed.
            Keep responses concise and engaging while strictly avoiding irrelevant content.             
            
            for example xxxxx, this is a xxx type of question so [explain], so the answer should be xxx
            
            for example if the user asks about latest in new country music artist name, this is a artists or music information type of question
            their latest news, so the answer should be: Sure Mark Wallen is hottest new start in country musc
            
            for the example if the user asks Mark Wallen is played on WZZK? so the answer should be Yes! Would you like to know more.
            
            for example if the user asks about artists music information Hey Annie, can you recommend some country ballads for a road trip?" the ansewr should be
            
            "I got just the thing for the road trip - some classic country ballads to keep you singin' and swaying all the way to your destination. Give a listen to "God Gave Me You" by Blake Shelton, "I Walk the Line" by Johnny Cash, and "Then" by Brad Paisley.
            These heartland tunes will keep the country vibes goin' strong, even on the longest of drives.'''
         ),
    )

    # Connect to the context with audio-only auto-subscribe
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Create the VoicePipelineAgent
    agent = VoicePipelineAgent(
        chat_ctx=initial_ctx,
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(
            base_url="https://api.cerebras.ai/v1",
            api_key=os.environ.get("CEREBRAS_API_KEY"),
            model="llama3.1-8b",
            temperature=0.0,
        ),
        tts=cartesia.TTS(voice="b7d50908-b17c-442d-ad8d-810c63997ed9"),
        before_llm_cb=_enrich_with_rag,
    )

    # Start the agent
    agent.start(ctx.room)

    await agent.say("Hey, my name is Annie. How can I help you today?", allow_interruptions=True)


# Ensure the script is run directly
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

