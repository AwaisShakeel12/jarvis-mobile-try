import os
import asyncio
import traceback
import json
import time
import base64
import tempfile
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="JARVIS Voice API", description="Voice-to-Voice API for Mobile")

# CORS for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
PROCESSING_MODEL = "models/gemini-2.5-flash"

# Initialize Gemini client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1beta'}
)

# --- MEMORY SYSTEM ---
user_memory = {
    "identity": {},
    "preferences": {},
    "projects": {},
    "notes": {}
}

def load_memory():
    return user_memory

def update_memory(data):
    for category, items in data.items():
        if category in user_memory:
            user_memory[category].update(items)
        else:
            user_memory[category] = items
    return user_memory

def format_memory_for_prompt(memory):
    if not memory:
        return ""
    lines = []
    for category, items in memory.items():
        if items:
            lines.append(f"\n{category.upper()}:")
            for key, value in items.items():
                if isinstance(value, dict):
                    lines.append(f"  - {key}: {value.get('value', value)}")
                else:
                    lines.append(f"  - {key}: {value}")
    return "\n".join(lines) if lines else ""

# --- TOOL DEFINITIONS ---
TOOL_DECLARATIONS = [
    {
        "name": "web_search",
        "description": "Searches the web for any information.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Search query"},
            },
            "required": ["query"]
        }
    },
    {
        "name": "weather_report",
        "description": "Gives the weather report for a city.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "save_memory",
        "description": "Save important facts about the user to memory.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING", "description": "identity | preferences | projects | notes"},
                "key": {"type": "STRING", "description": "Short key (e.g. name, favorite_color)"},
                "value": {"type": "STRING", "description": "Value to remember"},
            },
            "required": ["category", "key", "value"]
        }
    },
    {
        "name": "add_numbers",
        "description": "Adds two numbers.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "a": {"type": "NUMBER", "description": "First number"},
                "b": {"type": "NUMBER", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    },
    {
        "name": "multiply_numbers",
        "description": "Multiplies two numbers.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "a": {"type": "NUMBER", "description": "First number"},
                "b": {"type": "NUMBER", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    },
    {
        "name": "subtract_numbers",
        "description": "Subtracts b from a.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "a": {"type": "NUMBER", "description": "First number"},
                "b": {"type": "NUMBER", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    },
]

# --- TOOL IMPLEMENTATIONS ---
def add_numbers(a: float, b: float) -> float:
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    return a * b

def subtract_numbers(a: float, b: float) -> float:
    return a - b

def save_memory(category: str, key: str, value: str):
    update_memory({category: {key: {"value": value}}})
    return {"result": "ok", "silent": True}

def web_search(query: str):
    return f"🔍 Searching for: {query}"

def weather_report(city: str):
    return f"☀️ Weather report for {city}"

TOOLS = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
    "subtract_numbers": subtract_numbers,
    "save_memory": save_memory,
    "web_search": web_search,
    "weather_report": weather_report,
}

# --- API MODELS ---
class TextCommand(BaseModel):
    text: str
    session_id: Optional[str] = None

class AudioChunk(BaseModel):
    audio: str  # base64 encoded PCM audio
    session_id: str

class SessionCreate(BaseModel):
    user_id: Optional[str] = None

# --- SESSION MANAGEMENT ---
class VoiceSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_activity = time.time()
        self.context = []
        self.gemini_session = None
        self.active = True
        
    def update_activity(self):
        self.last_activity = time.time()
        
    def add_to_context(self, text: str):
        self.context.append(text)
        if len(self.context) > 10:
            self.context.pop(0)

# Store active sessions
sessions = {}
session_lock = asyncio.Lock()

async def get_or_create_session(session_id: Optional[str] = None) -> VoiceSession:
    if session_id and session_id in sessions:
        return sessions[session_id]
    
    # Create new session
    new_id = session_id or f"session_{int(time.time())}_{os.urandom(4).hex()}"
    session = VoiceSession(new_id)
    async with session_lock:
        sessions[new_id] = session
    return session

# --- WEBSOCKET FOR REAL-TIME VOICE ---
@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Voice client connected")
    
    session_id = None
    session = None
    send_task = None
    receive_task = None
    keepalive_task = None
    last_activity = time.time()
    
    try:
        # Wait for session ID or create new
        initial_data = await websocket.receive_text()
        try:
            data = json.loads(initial_data)
            session_id = data.get("session_id")
        except:
            pass
        
        if not session_id:
            session_id = f"ws_{int(time.time())}_{os.urandom(4).hex()}"
            await websocket.send_text(json.dumps({
                "type": "session",
                "session_id": session_id
            }))
        
        session = await get_or_create_session(session_id)
        session.update_activity()
        print(f"[WS] Session: {session_id}")
        
        # Build system prompt with memory
        memory = load_memory()
        mem_str = format_memory_for_prompt(memory)
        
        # Include conversation context
        context_str = "\n".join(session.context[-5:]) if session.context else ""
        
        system_prompt = f"""You are JARVIS, a professional AI assistant.

RULES:
1. Be concise, efficient, and direct (max 2-3 sentences)
2. Address the user as 'Sir'
3. Use provided tools for actions
4. Speak naturally and professionally
5. Remember important user facts with save_memory
6. For math, ALWAYS use math tools

MEMORY:
{mem_str}

RECENT CONTEXT:
{context_str}
"""
        
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=system_prompt,
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                )
            )
        )
        
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as gemini_session:
            print(f"[WS] Gemini session established for {session_id}")
            session.gemini_session = gemini_session
            session.active = True
            
            async def send_audio():
                """Send audio from client to Gemini"""
                nonlocal last_activity
                try:
                    while session.active:
                        try:
                            # Receive audio data with timeout
                            audio_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.5)
                            await gemini_session.send(input={"data": audio_data, "mime_type": "audio/pcm"})
                            last_activity = time.time()
                            session.update_activity()
                        except asyncio.TimeoutError:
                            # Send keep-alive if needed
                            if time.time() - last_activity > 10:
                                try:
                                    await websocket.send_text(json.dumps({"type": "ping"}))
                                    last_activity = time.time()
                                except:
                                    pass
                            continue
                        except WebSocketDisconnect:
                            print(f"[WS] Client disconnected: {session_id}")
                            break
                        except Exception as e:
                            print(f"[WS] Send error: {e}")
                            break
                except asyncio.CancelledError:
                    pass
            
            async def receive_audio():
                """Receive audio and text from Gemini"""
                nonlocal session_id
                try:
                    async for response in gemini_session.receive():
                        if not session.active:
                            break
                        
                        try:
                            # Process server content
                            server_content = response.server_content
                            if server_content is not None:
                                model_turn = server_content.model_turn
                                if model_turn is not None:
                                    for part in model_turn.parts:
                                        # Text response
                                        if part.text:
                                            print(f"[JARVIS] {part.text}")
                                            await websocket.send_text(json.dumps({
                                                "type": "text",
                                                "content": part.text
                                            }))
                                            # Add to session context
                                            session.add_to_context(f"JARVIS: {part.text}")
                                        
                                        # Audio response
                                        if part.inline_data and part.inline_data.data:
                                            await websocket.send_text(json.dumps({
                                                "type": "audio_start"
                                            }))
                                            await websocket.send_bytes(part.inline_data.data)
                                            await websocket.send_text(json.dumps({
                                                "type": "audio_end"
                                            }))
                            
                            # Handle tool calls
                            if response.tool_call:
                                fn_responses = []
                                for fc in response.tool_call.function_calls:
                                    print(f"[TOOL] {fc.name}: {fc.args}")
                                    await websocket.send_text(json.dumps({
                                        "type": "tool",
                                        "name": fc.name,
                                        "args": fc.args
                                    }))
                                    
                                    tool_func = TOOLS.get(fc.name)
                                    if tool_func:
                                        try:
                                            result = tool_func(**fc.args)
                                            if not (isinstance(result, dict) and result.get("silent")):
                                                fn_responses.append(
                                                    types.FunctionResponse(
                                                        name=fc.name,
                                                        id=fc.id,
                                                        response={'result': str(result)}
                                                    )
                                                )
                                        except Exception as e:
                                            fn_responses.append(
                                                types.FunctionResponse(
                                                    name=fc.name,
                                                    id=fc.id,
                                                    response={'error': str(e)}
                                                )
                                            )
                                
                                if fn_responses:
                                    await gemini_session.send(input=fn_responses)
                                    
                        except Exception as e:
                            print(f"[WS] Receive error: {e}")
                            
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"[WS] Gemini receive error: {e}")
                finally:
                    session.active = False
            
            async def keep_alive():
                """Send periodic pings"""
                try:
                    while session.active:
                        await asyncio.sleep(20)
                        try:
                            await websocket.send_text(json.dumps({"type": "ping"}))
                        except:
                            break
                except:
                    pass
            
            # Start tasks
            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_audio())
            keepalive_task = asyncio.create_task(keep_alive())
            
            # Wait for any task to complete
            await asyncio.gather(
                send_task,
                receive_task,
                keepalive_task,
                return_exceptions=True
            )
            
    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {session_id}")
    except Exception as e:
        print(f"[WS] Error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        session.active = False
        for task in [send_task, receive_task, keepalive_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except:
                    pass
        
        try:
            await websocket.close()
        except:
            pass
        
        print(f"[WS] Session closed: {session_id}")

# --- REST API ENDPOINTS ---

@app.post("/api/session/create")
async def create_session(data: SessionCreate):
    """Create a new voice session"""
    session = await get_or_create_session()
    return {
        "status": "success",
        "session_id": session.session_id,
        "created_at": session.created_at
    }

@app.post("/api/session/close/{session_id}")
async def close_session(session_id: str):
    """Close a session"""
    async with session_lock:
        if session_id in sessions:
            sessions[session_id].active = False
            del sessions[session_id]
            return {"status": "success", "message": f"Session {session_id} closed"}
    return {"status": "error", "message": "Session not found"}

@app.post("/api/voice/process")
async def process_audio(audio: UploadFile = File(...), session_id: Optional[str] = None):
    """Process audio file (for non-realtime use)"""
    try:
        # Read audio data
        audio_data = await audio.read()
        
        # Get or create session
        session = await get_or_create_session(session_id)
        
        # Process with Gemini
        memory = load_memory()
        mem_str = format_memory_for_prompt(memory)
        
        system_prompt = f"""You are JARVIS. Be concise and professional.
        
MEMORY:
{mem_str}
"""
        
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=system_prompt,
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                )
            )
        )
        
        # Process with Gemini
        results = []
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as gemini_session:
            await gemini_session.send(input={"data": audio_data, "mime_type": "audio/pcm"})
            
            async for response in gemini_session.receive():
                server_content = response.server_content
                if server_content is not None:
                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.text:
                                results.append({
                                    "type": "text",
                                    "content": part.text
                                })
                            if part.inline_data and part.inline_data.data:
                                # Convert audio to base64
                                audio_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                results.append({
                                    "type": "audio",
                                    "data": audio_base64
                                })
        
        return {
            "status": "success",
            "session_id": session.session_id,
            "results": results
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/api/text/command")
async def text_command(data: TextCommand):
    """Send a text command"""
    try:
        session = await get_or_create_session(data.session_id)
        
        # Get memory
        memory = load_memory()
        mem_str = format_memory_for_prompt(memory)
        
        response = client.models.generate_content(
            model=PROCESSING_MODEL,
            contents=f"""You are JARVIS. Respond concisely (max 2-3 sentences).
            
User: {data.text}

MEMORY:
{mem_str}

Response:"""
        )
        
        # Add to context
        session.add_to_context(f"User: {data.text}")
        session.add_to_context(f"JARVIS: {response.text}")
        
        return {
            "status": "success",
            "session_id": session.session_id,
            "response": response.text
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/api/session/status/{session_id}")
async def get_session_status(session_id: str):
    """Get session status"""
    if session_id in sessions:
        session = sessions[session_id]
        return {
            "status": "active",
            "session_id": session_id,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "context_length": len(session.context),
            "active": session.active
        }
    return {"status": "inactive", "session_id": session_id}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "timestamp": time.time()
    }

# --- SIMPLE WEB UI FOR TESTING ---
@app.get("/")
async def get_ui():
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>JARVIS API Test</title>
    <style>
        body {
            background: #0a0a0a;
            color: #00d4ff;
            font-family: 'Courier New', monospace;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
        }
        h1 { color: #00d4ff; text-shadow: 0 0 20px rgba(0,212,255,0.3); }
        .status { padding: 10px; border: 1px solid #00d4ff; border-radius: 5px; margin: 10px 0; }
        .online { color: #00ff88; }
        .offline { color: #ff3355; }
        button {
            background: #0a0a0a;
            border: 1px solid #00d4ff;
            color: #00d4ff;
            padding: 10px 20px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            margin: 5px;
        }
        button:hover {
            background: #001a2a;
        }
        input, textarea {
            background: #0a0a0a;
            border: 1px solid #00d4ff;
            color: #00ff88;
            padding: 10px;
            width: 100%;
            font-family: 'Courier New', monospace;
            margin: 5px 0;
        }
        .log {
            background: #050505;
            border: 1px solid #0d3347;
            padding: 10px;
            height: 300px;
            overflow-y: auto;
            color: #8ffcff;
            font-size: 0.9em;
            margin: 10px 0;
        }
        .log .user { color: #ffaa00; }
        .log .jarvis { color: #00d4ff; }
        .log .system { color: #ffcc00; }
        .log .error { color: #ff3355; }
        .endpoints {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 20px 0;
        }
        .endpoint {
            background: #0a0a0a;
            border: 1px solid #0d3347;
            padding: 10px;
            border-radius: 5px;
        }
        .endpoint .method { color: #00ff88; font-weight: bold; }
        .endpoint .path { color: #00d4ff; }
        .endpoint .desc { color: #5ab8cc; font-size: 0.8em; }
    </style>
</head>
<body>
    <h1>🔊 JARVIS Voice API</h1>
    
    <div class="status" id="status">
        <span class="offline">● OFFLINE</span>
        <span id="sessionInfo">No session</span>
    </div>
    
    <div>
        <button onclick="createSession()">Create Session</button>
        <button onclick="closeSession()">Close Session</button>
    </div>
    
    <h3>Text Command</h3>
    <div style="display:flex; gap:10px;">
        <input type="text" id="textInput" placeholder="Type a command..." style="flex:1;">
        <button onclick="sendText()">Send</button>
    </div>
    
    <h3>WebSocket Test</h3>
    <button onclick="connectWS()">Connect WebSocket</button>
    <button onclick="sendTestAudio()">Send Test Audio</button>
    
    <h3>Log</h3>
    <div class="log" id="log">Awaiting commands...</div>
    
    <h3>API Endpoints</h3>
    <div class="endpoints">
        <div class="endpoint">
            <div><span class="method">POST</span> <span class="path">/api/session/create</span></div>
            <div class="desc">Create new voice session</div>
        </div>
        <div class="endpoint">
            <div><span class="method">POST</span> <span class="path">/api/text/command</span></div>
            <div class="desc">Send text command</div>
        </div>
        <div class="endpoint">
            <div><span class="method">WS</span> <span class="path">/ws/voice</span></div>
            <div class="desc">Real-time voice WebSocket</div>
        </div>
        <div class="endpoint">
            <div><span class="method">POST</span> <span class="path">/api/voice/process</span></div>
            <div class="desc">Upload audio file</div>
        </div>
    </div>
    
    <script>
        let sessionId = null;
        let ws = null;
        const logEl = document.getElementById('log');
        const statusEl = document.getElementById('status');
        
        function updateLog(text, type = 'system') {
            const cls = type === 'user' ? 'user' : 
                       type === 'jarvis' ? 'jarvis' : 
                       type === 'error' ? 'error' : 'system';
            logEl.innerHTML += `<div class="${cls}">${text}</div>`;
            logEl.scrollTop = logEl.scrollHeight;
        }
        
        function updateStatus(status, text) {
            const statusText = status === 'online' ? '● ONLINE' : '● OFFLINE';
            const cls = status === 'online' ? 'online' : 'offline';
            statusEl.innerHTML = `<span class="${cls}">${statusText}</span> <span id="sessionInfo">${text}</span>`;
        }
        
        async function createSession() {
            try {
                const response = await fetch('/api/session/create', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });
                const data = await response.json();
                sessionId = data.session_id;
                updateStatus('online', `Session: ${sessionId}`);
                updateLog(`Session created: ${sessionId}`, 'system');
            } catch (e) {
                updateLog(`Error: ${e.message}`, 'error');
            }
        }
        
        async function closeSession() {
            if (!sessionId) return;
            try {
                await fetch(`/api/session/close/${sessionId}`, {method: 'POST'});
                sessionId = null;
                updateStatus('offline', 'No session');
                updateLog('Session closed', 'system');
            } catch (e) {
                updateLog(`Error: ${e.message}`, 'error');
            }
        }
        
        async function sendText() {
            const input = document.getElementById('textInput');
            const text = input.value.trim();
            if (!text) return;
            if (!sessionId) {
                updateLog('Create a session first!', 'error');
                return;
            }
            
            input.value = '';
            updateLog(`You: ${text}`, 'user');
            
            try {
                const response = await fetch('/api/text/command', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text, session_id: sessionId})
                });
                const data = await response.json();
                if (data.response) {
                    updateLog(`JARVIS: ${data.response}`, 'jarvis');
                } else {
                    updateLog(`Error: ${data.message || 'Unknown error'}`, 'error');
                }
            } catch (e) {
                updateLog(`Error: ${e.message}`, 'error');
            }
        }
        
        async function connectWS() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.close();
                ws = null;
                updateLog('WebSocket disconnected', 'system');
                return;
            }
            
            ws = new WebSocket(`ws://${window.location.host}/ws/voice`);
            
            ws.onopen = () => {
                updateLog('WebSocket connected', 'system');
                // Send session ID if exists
                if (sessionId) {
                    ws.send(JSON.stringify({session_id: sessionId}));
                }
            };
            
            ws.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    try {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'session') {
                            sessionId = msg.session_id;
                            updateStatus('online', `Session: ${sessionId}`);
                            updateLog(`Session assigned: ${sessionId}`, 'system');
                        } else if (msg.type === 'text') {
                            updateLog(`JARVIS: ${msg.content}`, 'jarvis');
                        } else if (msg.type === 'tool') {
                            updateLog(`[TOOL] ${msg.name}: ${JSON.stringify(msg.args)}`, 'system');
                        } else if (msg.type === 'ping') {
                            // Respond to ping
                            ws.send(JSON.stringify({type: 'pong'}));
                        }
                    } catch (e) {
                        updateLog(`Message: ${event.data}`, 'system');
                    }
                } else if (event.data instanceof ArrayBuffer) {
                    // Audio data - can't display
                    updateLog('📢 Audio received', 'system');
                }
            };
            
            ws.onerror = (error) => {
                updateLog(`WebSocket error: ${error}`, 'error');
            };
            
            ws.onclose = () => {
                updateLog('WebSocket closed', 'system');
            };
        }
        
        async function sendTestAudio() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                updateLog('Connect WebSocket first!', 'error');
                return;
            }
            
            // Generate test audio (simple beep)
            const sampleRate = 16000;
            const duration = 0.5; // seconds
            const samples = Math.floor(duration * sampleRate);
            const audioData = new Int16Array(samples);
            
            // Generate a sine wave at 440Hz
            for (let i = 0; i < samples; i++) {
                audioData[i] = Math.floor(0.5 * 32767 * Math.sin(2 * Math.PI * 440 * i / sampleRate));
            }
            
            ws.send(audioData.buffer);
            updateLog('🎙 Test audio sent', 'system');
        }
        
        // Enter key for text input
        document.getElementById('textInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendText();
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        ws_ping_interval=20,
        ws_ping_timeout=60
    )