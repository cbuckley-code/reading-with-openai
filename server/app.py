from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

load_dotenv()

app = FastAPI()
client = OpenAI()

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/passages/")
async def send_passage(level: int = 0):
    template = """You are a helpful english reading assistant who generates 10 word passages from books that are in 
    the mystery category for a user to read ONLY select mystery stories. A user will pass in their reading level on a 
    scale of 0 to 3 (0 indicating beginning English reading skills and 3 is a highly proficient reader). 
    You should generate a 10 word passage in JSON. ONLY return JSON, and nothing more following these rules.
    1. The passage can NOT be greater than 10 words ever.  
    2. Always return the source and author of the passage and specifically the book or publication. 
    3. NEVER include the source in the passage because the user is going to be asked to READ the passage to practice 
    reading.
    4. ALWAYS include your reason why YOU selected the passage for the user to read; what lead you to it.
    5. NEVER make up passages always use a published piece.
    6. Return JSON that is usable in a Javascript environment.
    """
    human_template = "{level}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI()
    response = chain.invoke({"level": level})

    print("returning an analysis - " + response.content)
    return response.content


@app.post("/upload-audio/")
async def compare_reading(audio: UploadFile = File(...), passage: str = ""):

    # set up some prompts to ask OpenAI to compare the given audio to the prompt the user was asked to read
    template = """You are an expert English Reading teacher that understands how to create a plan for individuals
    that want to learn how to read English.  You analyze an individuals results of reading a passage and when the 
    "passage" does not match the "reading" you provide suggestions and areas to practice for the individual. The 
    suggestions you provide should always come back as an array with an "area" and "suggestion". You only
    output JSON and ONLY JSON suitable for a http response and always include the original passage and the individual's 
    reading of that passage."""
    human_template = """passage: "{passage}"
    reading: "{reading}"
    """

    # Read the contents of the uploaded audio file
    contents = await audio.read()

    buffer = BytesIO(contents)
    buffer.name = audio.filename

    print("received file " + audio.filename + " for passage " + passage)

    # Process the contents as bytes and return the
    transcript = process_audio(buffer)

    # chain the calls to have OpenAI analyze the reading by the individual
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    chain = chat_prompt | ChatOpenAI()
    response = chain.invoke({"passage": passage, "reading": transcript})
    return response.content


def process_audio(audio_file):
    # Your audio processing logic here
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file)

    return transcript.text
