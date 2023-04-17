from typing import List
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from summary import summarize_and_translate

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize_and_translate", response_class=HTMLResponse)
async def summarize_and_translate_text(request: Request, text: str = Form(...)):
    translated_sentences = summarize_and_translate(text=text, target_lang="ne")
    output_text = " ".join(translated_sentences)
    return templates.TemplateResponse("output.html", {"request": request, "input_text": text, "output_text": output_text})
