from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agents.pipeline import build_agent

load_dotenv()

os.environ.setdefault("AGENTFORGE_MOCK_MODE", "true")

app = FastAPI(
    title="AgentForge Mock API",
    description="Browser API for generating AgentForge agent blueprints in mock mode.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentRequest(BaseModel):
    user_request: str


class AgentResponse(BaseModel):
    goal: str
    frequency: str
    output_type: str
    tools_needed: list[str]
    tool_configurations: list[dict[str, Any]]
    system_prompt: str
    workflow_steps: list[dict[str, Any]]
    memory_config: dict[str, Any]
    validation_result: dict[str, Any]
    mode: str


@app.get("/")
def home() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "AgentForge Mock API",
        "mode": os.getenv("AGENTFORGE_MOCK_MODE", "true"),
        "docs": "/docs",
        "ui": "/ui",
    }


@app.post("/agents/generate", response_model=AgentResponse)
def generate_agent(request: AgentRequest) -> AgentResponse:
    agent = build_agent()

    result = agent.invoke(
        {
            "user_request": request.user_request,
            "goal": "",
            "frequency": "",
            "output_type": "",
            "tools_needed": [],
            "tool_configurations": [],
            "system_prompt": "",
            "workflow_steps": [],
            "memory_config": {},
            "validation_result": {},
        }
    )

    return AgentResponse(
        goal=result.get("goal", ""),
        frequency=result.get("frequency", ""),
        output_type=result.get("output_type", ""),
        tools_needed=result.get("tools_needed", []),
        tool_configurations=result.get("tool_configurations", []),
        system_prompt=result.get("system_prompt", ""),
        workflow_steps=result.get("workflow_steps", []),
        memory_config=result.get("memory_config", {}),
        validation_result=result.get("validation_result", {}),
        mode=os.getenv("AGENTFORGE_MOCK_MODE", "true"),
    )


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AgentForge Mock UI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 40px auto;
            background: #f7f7f7;
            color: #222;
        }
        h1 {
            color: #111;
        }
        textarea {
            width: 100%;
            height: 100px;
            font-size: 16px;
            padding: 12px;
        }
        button {
            margin-top: 12px;
            padding: 12px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        pre {
            background: #111;
            color: #00ff90;
            padding: 20px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .card {
            background: white;
            padding: 24px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>AgentForge Mock UI</h1>
        <p>Enter an agent idea below. This runs in mock mode, so no OpenAI API key is needed.</p>

        <textarea id="request">Create an agent that sends me daily AI news alerts</textarea>
        <br>
        <button onclick="generateAgent()">Generate Agent Blueprint</button>

        <h2>Output</h2>
        <pre id="output">Waiting for input...</pre>
    </div>

    <script>
        async function generateAgent() {
            const userRequest = document.getElementById("request").value;
            const output = document.getElementById("output");

            output.textContent = "Generating...";

            try {
                const response = await fetch("/agents/generate", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        user_request: userRequest
                    })
                });

                const data = await response.json();
                output.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                output.textContent = "Error: " + error;
            }
        }
    </script>
</body>
</html>
"""
