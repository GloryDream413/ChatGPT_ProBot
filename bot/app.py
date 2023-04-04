import os
import logging
import pymongo
import asyncio
import traceback
import html
import json
import tempfile
import pydub
from pathlib import Path
from datetime import datetime, timedelta

import telegram
from telegram import Update, User, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice
from telegram.ext import (
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    PreCheckoutQueryHandler,
    JobQueue,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode, ChatAction

import cryptomus

from bot import config
from bot import database
from bot import openai_utils
from bot.payment import CryptomusPayment


# setup
db = database.Database()
logger = logging.getLogger(__name__)
user_semaphores = {}

HELP_MESSAGE = """<b>Commands</b>:
⚪ /retry – Regenerate last bot answer
⚪ /new – Start new dialog
⚪ /mode – Select chat mode
⚪ /balance – Show balance
⚪ /help – Show help

<b>Important note 1.</b> The longer your dialog, the more tokens are spent with each new message. To reset current dialog and start a new one, send /new command

<b>Important note 2.</b> Write 🇬🇧 English for a better quality of answers
"""


# --- Utility Functions --- #
def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


async def register_user(update: Update, context: CallbackContext, user: User):
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            update.message.chat_id,
            initial_token_balance=config.initial_token_balance,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name
        )
        db.start_new_dialog(user.id)

    if db.get_user_attribute(user.id, "current_dialog_id") is None:
        db.start_new_dialog(user.id)

    # token balance
    if not db.check_if_user_attribute_exists(user.id, "token_balance"):
        db.set_user_attribute(user.id, "token_balance", config.initial_token_balance)

    # openai engine
    if not db.check_if_user_attribute_exists(user.id, "openai_engine"):
        openai_engine = "chatgpt" if config.use_chatgpt_api else "gpt"
        db.set_user_attribute(user.id, "openai_engine", openai_engine)

    if user.id not in user_semaphores:
         user_semaphores[user.id] = asyncio.Semaphore(1)


async def is_previous_message_not_answered_yet(update: Update, context: CallbackContext):
     await register_user(update, context, update.message.from_user)

     user_id = update.message.from_user.id
     if user_semaphores[user_id].locked():
         text = "⏳ Please <b>wait</b> for a reply to the previous message"
         await update.message.reply_text(text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML)
         return True
     else:
        return False


async def check_if_user_has_enough_tokens(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id

    token_balance = db.get_user_attribute(user_id, "token_balance")
    if token_balance <= 0:
        await show_balance_handle(update, context)
        return False

    return True


async def send_user_message_about_n_added_tokens(context: CallbackContext, n_tokens_added: int, chat_id: int = None, user_id: int = None):
    if chat_id is None:
        if user_id is None:
            raise ValueError(f"chat_id and user_id can't be None simultaneously")
        chat_id = db.get_user_attribute(user_id, "chat_id")

    text = f"🟣 <b>{n_tokens_added}</b> tokens were successfully added to your balance!"
    await context.bot.send_message(chat_id, text, parse_mode=ParseMode.HTML)


async def notify_admins_about_successfull_payment(context: CallbackContext, payment_id: int):
    if config.admin_chat_id is not None:
        text = "🟣 Successfull payment:\n"

        payment_dict = db.payment_collection.find_one({"_id": payment_id})
        for key in ["amount", "currency", "product_key", "n_tokens_to_add", "payment_method", "user_id", "status"]:
            text += f"- {key}: <b>{payment_dict[key]}</b>\n"

        user_dict = db.user_collection.find_one({"_id": payment_dict["user_id"]})
        if user_dict["username"] is not None:
            text += f"- username: @{user_dict['username']}\n"

        # tag admins
        for admin_username in config.admin_usernames:
            if not admin_username.startswith("@"):
                admin_username = "@" + admin_username
            text += f"\n{admin_username}"
        
        await context.bot.send_message(config.admin_chat_id, text, parse_mode=ParseMode.HTML)


async def check_payment_status_job_fn(context: CallbackContext):
    job = context.job
    payment_id = job.data["payment_id"]
    payment_dict = db.payment_collection.find_one({"_id": payment_id})

    is_paid = False
    if payment_dict["payment_method_type"] == "cryptomus":
        cryptomus_payment_instance = CryptomusPayment(
            config.payment_methods["cryptomus"]["api_key"],
            config.payment_methods["cryptomus"]["merchant_id"]
        )

        is_paid = cryptomus_payment_instance.check_invoice_status(payment_id)
    else:
        context.job.schedule_removal()
        return

    if is_paid:
        db.set_payment_attribute(payment_id, "status", "paid")
        user_id = payment_dict["user_id"]

        n_tokens_to_add = payment_dict["n_tokens_to_add"]
        if not payment_dict["are_tokens_added"]:
            db.set_user_attribute(
                user_id,
                "token_balance",
                db.get_user_attribute(user_id, "token_balance") + n_tokens_to_add
            )
            db.set_payment_attribute(payment_id, "are_tokens_added", True)

            await send_user_message_about_n_added_tokens(context, n_tokens_to_add, chat_id=job.chat_id)
            await notify_admins_about_successfull_payment(context, payment_id)

        context.job.schedule_removal()


def run_repeating_payment_status_check(job_queue: JobQueue, payment_id: int, chat_id: int, how_long: int = 4000, interval: int = 30):
    job_queue.run_repeating(
        check_payment_status_job_fn,
        interval,
        first=0,
        last=how_long,
        name=str(payment_id),
        data={"payment_id": payment_id},
        chat_id=chat_id
    )


def run_not_expired_payment_status_check(job_queue: JobQueue, how_long: int = 4000, interval: int = 30):
    payment_ids = db.get_all_not_expried_payment_ids()
    for payment_id in payment_ids:
        user_id = db.get_payment_attribute(payment_id, "user_id")
        chat_id = db.get_user_attribute(user_id, "chat_id")
        run_repeating_payment_status_check(job_queue, payment_id, chat_id, how_long, interval)



# --- System Handles --- #
async def start_handle(update: Update, context: CallbackContext):
    await register_user(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)
    
    reply_text = "Hi! I'm <b>ChatGPT</b> bot implemented with GPT-4 OpenAI API 🤖\n\n"
    reply_text += HELP_MESSAGE

    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)

    async def show_chat_modes(context: CallbackContext):
        data = context.job.data
        await show_chat_modes_handle(data["update"], data["context"])
    context.job_queue.run_once(show_chat_modes, when=5, data={"update": update, "context": context})


async def help_handle(update: Update, context: CallbackContext):
    await register_user(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)



# --- Message Handles --- #
async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):
    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    await register_user(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return
    if not await check_if_user_has_enough_tokens(update, context): return

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    async with user_semaphores[user_id]:
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"Starting new dialog due to timeout (<b>{openai_utils.CHAT_MODES[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # send typing action
        await update.message.chat.send_action(action="typing")

        message = message or update.message.text
        dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
        use_chatgpt_api = db.get_user_attribute(user_id, "openai_engine") == "chatgpt"
        parse_mode = {
            "html": ParseMode.HTML,
            "markdown": ParseMode.MARKDOWN
        }[openai_utils.CHAT_MODES[chat_mode]["parse_mode"]]

        chatgpt_instance = openai_utils.ChatGPT(use_chatgpt_api=config.use_chatgpt_api)
        if config.enable_message_streaming:
            gen = chatgpt_instance.send_message_stream(message, dialog_messages=dialog_messages, chat_mode=chat_mode)
        else:
            answer, n_used_tokens, n_first_dialog_messages_removed = await chatgpt_instance.send_message(
                message,
                dialog_messages=dialog_messages,
                chat_mode=chat_mode
            )

            async def fake_gen():
                yield "finished", answer, n_used_tokens, n_first_dialog_messages_removed

            gen = fake_gen()

        # send message to user
        prev_answer = ""
        i = -1
        async for gen_item in gen:
            i += 1

            status = gen_item[0]
            if status == "not_finished":
                status, answer = gen_item
            elif status == "finished":
                status, answer, n_used_tokens, n_first_dialog_messages_removed = gen_item
            else:
                raise ValueError(f"Streaming status {status} is unknown")

            answer = answer[:4096]  # telegram message limit
            if i == 0:  # send first message (then it'll be edited if message streaming is enabled)
                try:                    
                    sent_message = await update.message.reply_text(answer, parse_mode=parse_mode)
                except telegram.error.BadRequest as e:
                    if str(e).startswith("Message must be non-empty"):  # first answer chunk from openai was empty
                        i = -1  # try again to send first message
                        continue
                    else:
                        sent_message = await update.message.reply_text(answer)
            else:  # edit sent message
                # update only when 100 new symbols are ready
                if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                    continue

                try:                    
                    await context.bot.edit_message_text(answer, chat_id=sent_message.chat_id, message_id=sent_message.message_id, parse_mode=parse_mode)
                except telegram.error.BadRequest as e:
                    if str(e).startswith("Message is not modified"):
                        continue
                    else:
                        await context.bot.edit_message_text(answer, chat_id=sent_message.chat_id, message_id=sent_message.message_id)

                await asyncio.sleep(0.01)  # wait a bit to avoid flooding
                
            prev_answer = answer

        # update user data
        new_dialog_message = {"user": message, "bot": answer, "date": datetime.now()}
        db.set_dialog_messages(
            user_id,
            db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
            dialog_id=None
        )

        db.set_user_attribute(user_id, "n_used_tokens", n_used_tokens + db.get_user_attribute(user_id, "n_used_tokens"))
        db.set_user_attribute(user_id, "token_balance", max(0, db.get_user_attribute(user_id, "token_balance") - n_used_tokens))

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
            else:
                text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def edited_message_handle(update: Update, context: CallbackContext):
    text = "🥲 Unfortunately, message <b>editing</b> is not supported"
    await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)


async def retry_handle(update: Update, context: CallbackContext):
    await register_user(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await update.message.reply_text("No message to retry 🤷‍♂️")
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # last message was removed from the context

    await message_handle(update, context, message=last_dialog_message["user"], use_new_dialog_timeout=False)


async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_user(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    db.start_new_dialog(user_id)
    await update.message.reply_text("Starting new dialog ✅")

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await update.message.reply_text(f"{openai_utils.CHAT_MODES[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)


async def voice_message_handle(update: Update, context: CallbackContext):
    await register_user(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    # check if user has enough tokens
    if not await check_if_user_has_enough_tokens(update, context):
        return

    voice = update.message.voice
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        voice_ogg_path = tmp_dir / "voice.ogg"

        # download
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive(voice_ogg_path)

        # convert to mp3
        voice_mp3_path = tmp_dir / "voice.mp3"
        pydub.AudioSegment.from_file(voice_ogg_path).export(voice_mp3_path, format="mp3")

        # transcribe
        with open(voice_mp3_path, "rb") as f:
            transcribed_text = await openai_utils.transcribe_audio(f)

    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    await message_handle(update, context, message=transcribed_text)

    # calculate spent dollars
    n_spent_dollars = voice.duration * (config.whisper_price_per_1_min / 60)

    # normalize dollars to tokens (it's very convenient to measure everything in a single unit)
    price_per_1000_tokens = config.chatgpt_price_per_1000_tokens if config.use_chatgpt_api else config.gpt_price_per_1000_tokens
    n_used_tokens = int(n_spent_dollars / (price_per_1000_tokens / 1000))
    
    db.set_user_attribute(user_id, "n_used_tokens", n_used_tokens + db.get_user_attribute(user_id, "n_used_tokens"))
    db.set_user_attribute(user_id, "token_balance", max(0, db.get_user_attribute(user_id, "token_balance") - n_used_tokens))



# --- Chat Mode Handles --- #
async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    keyboard = []
    for chat_mode, chat_mode_dict in openai_utils.CHAT_MODES.items():
        keyboard.append([InlineKeyboardButton(chat_mode_dict["name"], callback_data=f"set_chat_mode|{chat_mode}")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text("Select chat mode:", reply_markup=reply_markup)


async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)

    try:
        await query.edit_message_text(f"{openai_utils.CHAT_MODES[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass




# --- Payment Handles --- #
async def show_balance_handle(update: Update, context: CallbackContext):
    await register_user(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    token_balance = max(0, db.get_user_attribute(user_id, "token_balance"))
    n_used_tokens = db.get_user_attribute(user_id, "n_used_tokens")

    if token_balance > 0:
        text = f"🟢 You have <b>{token_balance}</b> tokens left\n"
    else:
        text = f"🔴 You have <b>NO</b> tokens left\n"
    text += f"You totally spent <b>{n_used_tokens}</b> tokens\n"
    
    reply_markup = InlineKeyboardMarkup([
        [InlineKeyboardButton("🥑 Get Tokens", callback_data=f"show_payment_methods")],
    ])

    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def show_payment_methods_handle(update: Update, context: CallbackContext):
    await register_user(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    text = "Choose <b>payment method</b>:"

    buttons = []
    for payment_method_key, payment_method_values in config.payment_methods.items():
        button = InlineKeyboardButton(
            payment_method_values["name"],
            callback_data=f"show_products|{payment_method_key}"
        )
        buttons.append([button])
    reply_markup = InlineKeyboardMarkup(buttons)

    await context.bot.send_message(
        update.callback_query.message.chat.id,
        text,
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )


async def show_products_handle(update: Update, context: CallbackContext):
    await register_user(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, payment_method_key = query.data.split("|")

    text = "How many <b>tokens</b> do you want to buy?"

    product_keys = config.payment_methods[payment_method_key]["product_keys"]
    buttons = []
    for product_key in product_keys:
        product = config.products[product_key]
        button = InlineKeyboardButton(
            product["title_on_button"],
            callback_data=f"send_invoice|{payment_method_key}|{product_key}"
        )
        buttons.append([button])
    reply_markup = InlineKeyboardMarkup(buttons)

    await context.bot.send_message(
        update.callback_query.message.chat.id,
        text,
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )


async def send_invoice_handle(update: Update, context: CallbackContext):
    await register_user(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, payment_method_key, product_key = query.data.split("|")
    product = config.products[product_key]
    payment_method_type = config.payment_methods[payment_method_key]["type"]

    payment_id = db.get_new_unique_payment_id()

    if payment_method_type == "telegram_payments":
        chat_id = update.callback_query.message.chat.id

        # save in database
        db.add_new_payment(
            payment_id=payment_id,
            payment_method=payment_method_key,
            payment_method_type=payment_method_type,
            product_key=product_key,
            user_id=user_id,
            amount=product["price"],
            currency=product["currency"],
            status="not_paid",
            invoice_url="",
            expired_at=datetime.now() + timedelta(hours=1),
            n_tokens_to_add=product["n_tokens_to_add"]
        )

        # create invoice
        payload = f"{payment_id}"
        prices = [LabeledPrice(product["title"], int(product["price"] * 100))]

        photo_url = None
        if "photo_url" in product and len(product["photo_url"]) > 0:
            photo_url = product["photo_url"]

        # send invoice
        await context.bot.send_invoice(
            chat_id,
            product["title"],
            product["description"],
            payload,
            config.payment_methods[payment_method_key]["token"],
            product["currency"],
            prices,
            photo_url=photo_url
        )
    elif payment_method_type == "cryptomus":
        # create invoice
        cryptomus_payment_instance = CryptomusPayment(
            config.payment_methods[payment_method_key]["api_key"],
            config.payment_methods[payment_method_key]["merchant_id"]
        )

        invoice_url, status, expired_at = cryptomus_payment_instance.create_invoice(
            payment_id,
            product["price"],
            product["currency"]
        )

        # save in database
        db.add_new_payment(
            payment_id=payment_id,
            payment_method=payment_method_key,
            payment_method_type=payment_method_type,
            product_key=product_key,
            user_id=user_id,
            amount=product["price"],
            currency=product["currency"],
            status=status,
            invoice_url=invoice_url,
            expired_at=expired_at,
            n_tokens_to_add=product["n_tokens_to_add"]
        )

        # run status check polling
        run_repeating_payment_status_check(context.job_queue, payment_id, update.callback_query.message.chat.id, how_long=4000, interval=30)

        # send invoice
        text = f"🔗 Here is: <a href=\"{invoice_url}\">your invoice</a> (<b>{product['n_tokens_to_add']}</b> tokens)\n\n"
        text += "You have 1 hour to pay it before it expires. If you are facing any problems, write to ."
        await context.bot.send_message(update.callback_query.message.chat.id, text, parse_mode=ParseMode.HTML)
    else:
        raise ValueError(f"Unknown payment method: {payment_method_type}")


async def pre_checkout_handle(update: Update, context: CallbackContext):
    await register_user(update.pre_checkout_query, context, update.pre_checkout_query.from_user)
    user_id = update.pre_checkout_query.from_user.id

    query = update.pre_checkout_query
    await query.answer(ok=True)


async def successful_payment_handle(update: Update, context: CallbackContext):
    await register_user(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    chat_id = db.get_user_attribute(user_id, "chat_id")
    
    payment_id = int(update.message.successful_payment.invoice_payload)
    payment_dict = db.payment_collection.find_one({"_id": payment_id})

    # update payment in database
    db.set_payment_attribute(payment_id, "status", "paid")

    n_tokens_to_add = payment_dict["n_tokens_to_add"]
    if not payment_dict["are_tokens_added"]:
        db.set_user_attribute(
            user_id,
            "token_balance",
            db.get_user_attribute(user_id, "token_balance") + n_tokens_to_add
        )
        db.set_payment_attribute(payment_id, "are_tokens_added", True)

        # send messages
        await send_user_message_about_n_added_tokens(context, n_tokens_to_add, chat_id=chat_id)
        await notify_admins_about_successfull_payment(context, payment_id)


# --- Admin Handles --- #
async def add_tokens_handle(update: Update, context: CallbackContext):
    await register_user(update, context, update.message.from_user)

    username_or_user_id, n_tokens_to_add = context.args
    n_tokens_to_add = int(n_tokens_to_add)

    try:
        user_id = int(username_or_user_id)
        user_dict = db.user_collection.find_one({"_id": user_id})
    except:
        username = username_or_user_id
        user_dict = db.user_collection.find_one({"username": username})

    if user_dict is None:
        text = f"Username or user_id <b>{username_or_user_id}</b> not found in DB"
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        return

    # add tokens
    db.set_user_attribute(user_dict["_id"], "token_balance", db.get_user_attribute(user_dict["_id"], "token_balance") + n_tokens_to_add)

    # send message to user
    await send_user_message_about_n_added_tokens(context, n_tokens_to_add, chat_id=user_dict["chat_id"])

    # send message to admin
    text = f"🟣 <b>{n_tokens_to_add}</b> tokens were successfully added to <b>{username_or_user_id}</b> balance!"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        

async def switch_openai_engine_handle(update: Update, context: CallbackContext):
    await register_user(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    current_openai_engine = db.get_user_attribute(user_id, "openai_engine")
    if current_openai_engine == "chatgpt":
        new_openai_engine = "gpt"
    elif current_openai_engine == "gpt":
        new_openai_engine = "chatgpt"
    else:
        raise ValueError(f"Unknown OpenAI engine: {current_openai_engine}")

    db.set_user_attribute(user_id, "openai_engine", new_openai_engine)

    text = f"Switched to {new_openai_engine} engine ✅"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def user_info_handle(update: Update, context: CallbackContext):
    await register_user(update, context, update.message.from_user)
    user = update.message.from_user

    text = "User info:\n"
    text += f"- <b>user_id</b>: {user.id}\n"
    text += f"- <b>username</b>: {user.username}\n"
    text += f"- <b>chat_id</b>: {update.message.chat_id}\n"

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    if config.admin_chat_id is not None:
        await context.bot.send_message(config.admin_chat_id, text, parse_mode=ParseMode.HTML)


async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        await context.bot.send_message(update.effective_chat.id, "🥲 Something went wrong...\nPlease <b>try again later</b> or contact !", parse_mode=ParseMode.HTML)

        if config.admin_chat_id is not None:
            # collect error message
            tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
            tb_string = "".join(tb_list)
            update_str = update.to_dict() if isinstance(update, Update) else str(update)
            message = (
                f"An exception was raised while handling an update\n"
                f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
                "</pre>\n\n"
                f"<pre>{html.escape(tb_string)}</pre>"
            )

            # split text into multiple messages due to 4096 character limit
            for message_chunk in split_text_into_chunks(message, 4096):
                try:
                    await context.bot.send_message(config.admin_chat_id, message_chunk, parse_mode=ParseMode.HTML)
                except telegram.error.BadRequest:
                    # answer has invalid characters, so we send it without parse_mode
                    await context.bot.send_message(config.admin_chat_id, message_chunk)
    except:
        await context.bot.send_message(update.effective_chat.id, "🥲 Some error in error handler... Contact ")


def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=3))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .build()
    )

    # check not expired payments
    run_not_expired_payment_status_check(application.job_queue)
    
    # add handlers
    if len(config.allowed_telegram_usernames) == 0:
        user_filter = filters.ALL
    else:
        user_filter = filters.User(username=config.allowed_telegram_usernames)

    if len(config.admin_usernames) == 0:
        raise ValueError("You must specify at least 1 admin username in config")
    admin_filter = filters.User(username=config.admin_usernames)

    # system
    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))

    # message
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(CommandHandler("new", new_dialog_handle, filters=user_filter))
    application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))
    
    # chat mode
    application.add_handler(CommandHandler("mode", show_chat_modes_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode"))

    # payment
    application.add_handler(CommandHandler("balance", show_balance_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(show_payment_methods_handle, pattern="^show_payment_methods$"))
    application.add_handler(CallbackQueryHandler(show_products_handle, pattern="^show_products"))
    application.add_handler(CallbackQueryHandler(send_invoice_handle, pattern="^send_invoice"))

    application.add_handler(PreCheckoutQueryHandler(pre_checkout_handle))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT & user_filter, successful_payment_handle))

    # admin
    application.add_handler(CommandHandler("add_tokens", add_tokens_handle, filters=admin_filter))
    application.add_handler(CommandHandler("switch_openai_engine", switch_openai_engine_handle, filters=user_filter))   
    application.add_handler(CommandHandler("info", user_info_handle, filters=user_filter))   
    application.add_error_handler(error_handle)
    
    # start the bot
    application.run_polling()
