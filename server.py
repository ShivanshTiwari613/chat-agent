# filepath: server.py

import os
import uuid
import json
import asyncio
import threading
from typing import List, Any, cast
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.agent.llm_engine import AgentEngine
from app.utils.logger import logger
from app.utils.database import init_db, SessionLocal, SessionRecord, MessageRecord, FileRegistryRecord
from app.utils.title_generator import generate_chat_title

app = Flask(__name__)
CORS(app)

STORAGE_BASE = "persistent_storage"
os.makedirs(STORAGE_BASE, exist_ok=True)
init_db()

sessions: dict[str, AgentEngine] = {}
sessions_lock = threading.Lock()

def get_or_create_engine(plan_id: str) -> AgentEngine:
    """Returns engine instantly. Heavy rehydration is deferred to the async stream."""
    with sessions_lock:
        if plan_id in sessions:
            return sessions[plan_id]

        db = SessionLocal()
        try:
            engine = AgentEngine(plan_id=plan_id)
            session_rec = db.query(SessionRecord).filter_by(plan_id=plan_id).first()
            
            if session_rec:
                # Load history metadata only
                history_recs = db.query(MessageRecord).filter_by(plan_id=plan_id).order_by(MessageRecord.created_at.asc()).all()
                rehydrated: List[BaseMessage] = []
                for m in history_recs:
                    if str(m.role) == "user": 
                        rehydrated.append(HumanMessage(content=str(m.content)))
                    else: 
                        rehydrated.append(AIMessage(content=str(m.content)))
                engine.internal_history = rehydrated
            else:
                db.add(SessionRecord(plan_id=plan_id))
                db.commit()

            sessions[plan_id] = engine
            return engine
        finally:
            db.close()

def format_chat_history(history_data: list) -> List[BaseMessage]:
    formatted: List[BaseMessage] = []
    for msg in (history_data or []):
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role == "user": 
            formatted.append(HumanMessage(content=content))
        elif role == "assistant": 
            formatted.append(AIMessage(content=content))
    return formatted

@app.route("/sessions", methods=["GET"])
def list_sessions():
    db = SessionLocal()
    try:
        recs = db.query(SessionRecord).order_by(SessionRecord.created_at.desc()).all()
        return jsonify([{
            "plan_id": str(r.plan_id),
            "title": str(r.title or "New Conversation"),
            "created_at": r.created_at.isoformat()
        } for r in recs])
    finally:
        db.close()

@app.route("/upload", methods=["POST"])
def upload_files():
    plan_id = str(request.form.get("plan_id", ""))
    files = request.files.getlist("files")
    if not plan_id or not files: return jsonify({"error": "Missing data"}), 400

    try:
        engine = get_or_create_engine(plan_id)
        db = SessionLocal()
        session_dir = os.path.join(STORAGE_BASE, plan_id)
        os.makedirs(session_dir, exist_ok=True)

        uploaded_paths = []
        for file in files:
            f_name = str(file.filename or f"upload_{uuid.uuid4().hex}")
            path = os.path.join(session_dir, f_name)
            file.save(path)
            uploaded_paths.append(path)
            if not db.query(FileRegistryRecord).filter_by(plan_id=plan_id, filename=f_name).first():
                db.add(FileRegistryRecord(plan_id=plan_id, filename=f_name, namespace="pending"))
        db.commit()
        db.close()

        # Run indexing in a one-off loop to avoid blocking Flask
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(engine.add_files(uploaded_paths))
        finally:
            loop.close()
            
        return jsonify({"success": True, "filenames": [f.filename for f in files]})
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data: return jsonify({"error": "No JSON"}), 400
    plan_id = str(data.get("plan_id", ""))
    user_input = str(data.get("message", ""))
    history_raw = data.get("chat_history", [])

    # Instant response from get_or_create_engine
    engine = get_or_create_engine(plan_id)
    
    db = SessionLocal()
    db.add(MessageRecord(plan_id=plan_id, role="user", content=user_input))
    db.commit()
    db.close()

    def event_stream():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_chat():
            # 1. Background Rehydration: Only happens if index is empty
            if not engine.file_index.chunks:
                db_inner = SessionLocal()
                file_recs = db_inner.query(FileRegistryRecord).filter_by(plan_id=plan_id).all()
                db_inner.close()
                
                paths = [os.path.join(STORAGE_BASE, plan_id, str(f.filename)) for f in file_recs]
                paths = [p for p in paths if os.path.exists(p)]
                if paths:
                    # Inform UI that we are working on the index
                    from app.api.schema import AgentEvent
                    yield f"data: {AgentEvent(type='status', label='INDEXING', details='Rehydrating intelligence from disk...').model_dump_json()}\n\n"
                    await engine.add_files(paths)

            # 2. Lazy Initialization of Tools/Sandbox
            await engine.initialize()

            history = format_chat_history(history_raw)
            final_response = ""
            
            async for event in engine.chat(user_input, chat_history=history):
                if event.type == "result": 
                    final_response = str(event.content)
                # FIX: use model_dump_json() instead of .json()
                yield f"data: {event.model_dump_json()}\n\n"

            if final_response:
                db_end = SessionLocal()
                try:
                    db_end.add(MessageRecord(plan_id=plan_id, role="assistant", content=final_response))
                    session_rec = db_end.query(SessionRecord).filter_by(plan_id=plan_id).first()
                    if session_rec and not getattr(session_rec, 'title'):
                        new_title = await generate_chat_title(user_input, final_response)
                        cast(Any, session_rec).title = str(new_title)
                    db_end.commit()
                finally:
                    db_end.close()

        gen = run_chat()
        try:
            while True:
                try:
                    yield loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration: break
        except Exception as e:
            logger.error(f"SSE error: {e}")
            from app.api.schema import AgentEvent
            err_ev = AgentEvent(type='error', details=str(e))
            yield f"data: {err_ev.model_dump_json()}\n\n"
        finally:
            loop.close()

    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/session/<plan_id>", methods=["DELETE"])
def close_session(plan_id):
    p_id = str(plan_id)
    with sessions_lock:
        if p_id in sessions:
            engine = sessions.pop(p_id)
            loop = asyncio.new_event_loop()
            try: loop.run_until_complete(engine.cleanup())
            finally: loop.close()
            return jsonify({"success": True})
    return jsonify({"success": False}, 404)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)