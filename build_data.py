import asyncio
import pickle
import uuid
import os
import aiohttp
from livekit.agents import tokenize
from tqdm import tqdm
from livekit.plugins import deepgram, openai, rag, silero, cartesia

# Set your API key here
os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Embedding dimension for text-embedding-3-small
embeddings_dimension = 1536

# Read raw data with UTF-8 encoding
with open("raw_data.txt", "r", encoding="utf-8") as file:
    raw_data = file.read()


def split_text_into_chunks(text, chunk_size=2000):
    """Split the input text into chunks of the specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


async def _create_embeddings(input: str, http_session: aiohttp.ClientSession) -> dict:
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": input,
        "model": "text-embedding-3-small",
    }

    try:
        async with http_session.post(url, json=payload, headers=headers) as response:
            response_json = await response.json()
            print("Full Response:", response_json)  # Debugging output
            return response_json["data"][0]
    except Exception as e:
        print(f"Error while creating embeddings for input: {input[:100]}... | Error: {e}")
        return None


async def main() -> None:
    async with aiohttp.ClientSession() as http_session:
        idx_builder = rag.annoy.IndexBuilder(f=embeddings_dimension, metric="angular")

        paragraphs_by_uuid = {}
        chunks = split_text_into_chunks(raw_data, chunk_size=2000)

        for chunk in chunks:
            p_uuid = uuid.uuid4()
            paragraphs_by_uuid[p_uuid] = chunk

        for p_uuid, paragraph in tqdm(paragraphs_by_uuid.items()):
            resp = await _create_embeddings(paragraph, http_session)
            if resp is not None:
                idx_builder.add_item(resp["embedding"], p_uuid)
            else:
                print(f"Skipping paragraph with UUID {p_uuid} due to an error.")

        # Build and save the index
        idx_builder.build()
        idx_builder.save("vdb_data")

        # Save paragraph data with pickle
        with open("my_data.pkl", "wb") as f:
            pickle.dump(paragraphs_by_uuid, f)


if __name__ == "__main__":
    asyncio.run(main())

