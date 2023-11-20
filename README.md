# Welcome to Reading with OpenAI


The project is set up into two mini projects the server and client.
The server folder holds all the python APIs and the client folder has
the SvelteKit UI that interfaces with it.

## Server
To get it running update the .env file in the server directory
with you OpenAI key then run the following command.

> `uvicorn app:app --reload`


## Client
The client app is a SvelteKit app (and a real simple one at that).
To get it fired up make sure the server is running ;) then type.

> ' npm run dev -- --open'

## Hope you enjoy it!
I do want to thank Jason Zhou his YouTube videos sparked my interest
and his code sample here https://github.com/JayZeeDesign are great references!