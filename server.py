# filepath: server.py

import os
import uuid
import json
import asyncio
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage

from app.agent.llm_engine import AgentEngine
from app.utils.logger import logger
from app.api.schema import ChatRequest

app = Flask(__name__)
CORS(app)  # Enable CORS for UI integration

# Global session store: { plan_id: AgentEngine }
sessions: dict[str, AgentEngine] = {}

def get_or_create_engine(plan_id: str) -> AgentEngine:
    """Retrieves an existing engine or initializes a new one for the session."""
    if plan_id not in sessions:
        logger.info(f"Creating new engine session for ID: {plan_id}")
        sessions[plan_id] = AgentEngine(plan_id=plan_id)
    return sessions[plan_id]

def format_chat_history(history_data: list) -> list:
    """Converts raw JSON history into LangChain message objects."""
    formatted = []
    if not history_data:
        return formatted
    
    for msg in history_data:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            formatted.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted.append(AIMessage(content=content))
    return formatted

@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Endpoint to upload files (Images, PDFs, Code, ZIPs).
    Expects multipart/form-data with 'plan_id' and 'files'.
    """
    plan_id = request.form.get("plan_id")
    if not plan_id:
        return jsonify({"success": False, "error": "Missing plan_id"}), 400
    
    files = request.files.getlist("files")
    if not files:
        return jsonify({"success": False, "error": "No files provided"}), 400

    engine = get_or_create_engine(plan_id)
    
    uploaded_paths = []
    temp_dir = f"temp_uploads/{plan_id}"
    os.makedirs(temp_dir, exist_ok=True)

    for file in files:
        filename = file.filename or f"upload_{uuid.uuid4().hex}"
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        uploaded_paths.append(file_path)

    async def process_upload():
        # Ensure engine is initialized (sandbox started)
        if not engine.agent_executor:
            await engine.initialize()
        # Process files: Extraction -> Vision Analysis -> Indexing -> Sandbox Mirroring
        await engine.add_files(uploaded_paths)

    try:
        # Run the async processing logic
        asyncio.run(process_upload())
        
        # Cleanup local temp files after indexing
        for p in uploaded_paths:
            if os.path.exists(p): os.remove(p)
            
        return jsonify({
            "success": True, 
            "message": f"Successfully indexed {len(uploaded_paths)} files.",
            "filenames": [f.filename for f in files]
        })
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """
    Streaming Chat Endpoint (SSE).
    Accepts JSON: { "message": str, "plan_id": str, "chat_history": [] }
    """
    data = request.json
    plan_id = data.get("plan_id")
    user_input = data.get("message")
    history_raw = data.get("chat_history", [])

    if not plan_id or not user_input:
        return jsonify({"error": "Missing plan_id or message"}), 400

    engine = get_or_create_engine(plan_id)

    def event_stream():
        """Generator for Server-Sent Events."""
        # Create a new event loop for this thread to handle the async engine
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_chat():
            if not engine.agent_executor:
                await engine.initialize()
            
            # Format history from request - engine.chat will use this to seed internal memory
            history = format_chat_history(history_raw)
            
            async for event in engine.chat(user_input, chat_history=history):
                # Yield the structured JSON as an SSE message
                yield f"data: {event.json()}\n\n"

        gen = run_chat()
        try:
            while True:
                try:
                    # Drive the async generator
                    yield loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    break
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'details': str(e)})}\n\n"
        finally:
            loop.close()

    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/session/<plan_id>", methods=["DELETE"])
def close_session(plan_id):
    """Gracefully shuts down the sandbox and clears the engine from memory."""
    if plan_id in sessions:
        engine = sessions.pop(plan_id)
        try:
            # We use a proper way to run the async cleanup
            loop = asyncio.new_event_loop()
            loop.run_until_complete(engine.cleanup())
            loop.close()
        except Exception as e:
            logger.error(f"Cleanup error for {plan_id}: {e}")
        return jsonify({"success": True, "message": f"Session {plan_id} closed."})
    return jsonify({"success": False, "message": "Session not found."}), 404

if __name__ == "__main__":
    # Ensure local upload directory exists
    os.makedirs("temp_uploads", exist_ok=True)
    
    # Run Flask App
    # threaded=True is important for handling multiple SSE streams
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)