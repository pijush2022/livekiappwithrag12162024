# livekiappwithrag12162024


1. First Set an openai api key on both the python file
2. Create an Virtual environment with this commnad  `python -m venv venv`
3. Acivate the Venv environment with this commnad `venv\Scripts\activate` this isthe windows Commnand
4. Installed the requiremnents.txt with this commnad `python -r requirements.txt`
5. Create a .env file kept all the api key like 
        `LIVEKIT_URL="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"`
        `LIVEKIT_API_KEY="xxxxxxxxxxxxxxxxxxxxxx"`
        `LIVEKIT_API_SECRET="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"`
        `DEEPGRAM_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"`
        `CARTESIA_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"`
        `CEREBRAS_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"`
        `OPENAI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"`
6.Then run first the `build_data.py` for embeddings with command `python build_data.py`
7.After that Finally run the assistance.py with commnad with command `python assistance.py`
8.Then open the ivekit Playground with log in and then select the project and then clicked on connect and this is the playground link `https://agents-playground.livekit.io/`

Thats it.....
