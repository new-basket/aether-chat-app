# app.py (ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ)
from dotenv import load_dotenv
import os
import traceback
import json

load_dotenv()

from flask import Flask, render_template, request, Response
from flask_sqlalchemy import SQLAlchemy
from google import genai
from google.genai import types
from google.api_core import exceptions as google_api_exceptions

app = Flask(__name__)

# --- КОНФИГУРАЦИЯ БАЗЫ ДАННЫХ ---
db_url = os.environ.get('DATABASE_URL')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

if not db_url:
    print("ПРЕДУПРЕЖДЕНИЕ: DATABASE_URL не найден. Используется временная база SQLite.")
    db_url = "sqlite:///temp_chat.db"

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- КОНФИГУРАЦИЯ GEMINI ---
API_KEY_CONFIGURED = False
GEMINI_CLIENT = None
MODEL_NAME_DEFAULT = "gemini-2.5-flash-preview-05-20"
MAX_HISTORY_LENGTH = 20

SYSTEM_INSTRUCTION = {
    'role': 'user', 'parts': [{'text': "You will now adopt a new persona..."}]
}
SYSTEM_RESPONSE = {
    'role': 'model', 'parts': [{'text': "Эфир активен. Ожидаю приказа."}]
}

# --- МОДЕЛЬ ДАННЫХ ДЛЯ ИСТОРИИ ЧАТА ---
class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    user_id = db.Column(db.String(100), primary_key=True)
    history = db.Column(db.JSON, nullable=False)

    def __repr__(self):
        return f'<ChatSession for {self.user_id}>'

with app.app_context():
    db.create_all()
    print("Таблицы базы данных проверены/созданы.")

# Инициализация Gemini
try:
    api_key_from_env = os.environ.get("GOOGLE_API_KEY")
    if not api_key_from_env:
        print("ПРЕДУПРЕЖДЕНИЕ: GOOGLE_API_KEY не найден в переменных окружения.")
    else:
        GEMINI_CLIENT = genai.Client(api_key=api_key_from_env)
        API_KEY_CONFIGURED = True
        print(f"Клиент Google GenAI успешно инициализирован.")
except Exception as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА при инициализации клиента: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def handle_chat():
    if not API_KEY_CONFIGURED or not GEMINI_CLIENT:
        return Response("Ошибка: Чат-сервис не сконфигурирован.", status=503)

    try:
        req_data = request.json
        user_message_text = req_data.get("message")
        user_id = req_data.get("user_id")

        if not all([user_message_text, user_id]):
            return Response("Ошибка: В запросе отсутствуют message или user_id.", status=400)
        
        session = ChatSession.query.get(user_id)

        if not session:
            print(f"Создание новой сессии в БД для user_id: {user_id}")
            initial_history = [SYSTEM_INSTRUCTION, SYSTEM_RESPONSE]
            new_session = ChatSession(user_id=user_id, history=initial_history)
            db.session.add(new_session)
            db.session.commit()
            history = initial_history
        else:
            history = session.history

        history.append({'role': 'user', 'parts': [{'text': user_message_text}]})

        if len(history) > MAX_HISTORY_LENGTH:
            history = [SYSTEM_INSTRUCTION, SYSTEM_RESPONSE] + history[-MAX_HISTORY_LENGTH+2:]
            print(f"История для '{user_id}' обрезана.")

        print(f"Длина истории для '{user_id}': {len(history)} сообщений.")
        
        tools_config = [types.Tool(google_search=types.GoogleSearch())]
        generate_config = types.GenerateContentConfig(
            tools=tools_config,
            safety_settings=[
                types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
            ]
        )

        # --- ИЗМЕНЕНИЕ ВНУТРИ ЭТОЙ ФУНКЦИИ ---
        def generate_response_chunks():
            full_bot_response = ""
            try:
                stream = GEMINI_CLIENT.models.generate_content_stream(
                    model=MODEL_NAME_DEFAULT, contents=history, config=generate_config
                )
                for chunk in stream:
                    if text_part := chunk.text:
                        full_bot_response += text_part
                        yield text_part
                    if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                         error_msg = f"\n[СИСТЕМНОЕ УВЕДОМЛЕНИЕ: Запрос заблокирован: {chunk.prompt_feedback.block_reason.name}]"
                         print(f"!!! Блокировка для '{user_id}': {chunk.prompt_feedback.block_reason.name}")
                         yield error_msg
                         return

                if full_bot_response:
                    # --- ИЗМЕНЕНИЕ ЗДЕСЬ! ---
                    # Мы оборачиваем операции с базой данных в app_context,
                    # чтобы они работали вне основного потока запроса.
                    with app.app_context():
                        history.append({'role': 'model', 'parts': [{'text': full_bot_response}]})
                        current_session = ChatSession.query.get(user_id)
                        if current_session:
                            current_session.history = history
                            db.session.commit()
                            print(f"История для '{user_id}' успешно сохранена в БД.")
                    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            except Exception as e:
                print(f"!!! Ошибка во время стриминга для '{user_id}': {e}")
                traceback.print_exc()
                yield "Извините, произошла внутренняя ошибка при генерации ответа."

        return Response(generate_response_chunks(), mimetype='text/plain')

    except Exception as e:
        print(f"!!! Непредвиденная ошибка в /chat: {e}")
        traceback.print_exc()
        return Response(f"Внутренняя ошибка сервера: {str(e)}", status=500)

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
