import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from app.utils.logger import logger
from config.settings import settings


async def generate_chat_title(user_input: str, ai_response: str) -> str:
    """
    Analyzes the first exchange of a chat and generates a concise title.
    """
    try:
        # We use a lower temperature for consistent, descriptive titles
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_NAME,
            temperature=0.3,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        prompt = (
            "You are a helpful assistant that generates short, descriptive titles for chat conversations. "
            "Based on the following user message and AI response, create a 3-to-6 word title that captures the core topic. "
            "Return ONLY the title text. Do not use quotes or a period at the end.\n\n"
            f"USER: {user_input}\n"
            f"ASSISTANT: {ai_response}"
        )

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        title = str(response.content).strip()

        # Strip common undesirable characters if the LLM ignores instructions
        title = title.replace('"', "").replace("'", "").strip()

        logger.info(f"Generated title: {title}")
        return title
    except Exception as e:
        logger.error(f"Title generation failed: {e}")
        # Fallback to a truncated version of the user input if LLM fails
        return user_input[:30] + "..." if len(user_input) > 30 else user_input
