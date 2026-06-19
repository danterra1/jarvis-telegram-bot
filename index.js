import 'dotenv/config';
import TelegramBot from 'node-telegram-bot-api';
import Anthropic from '@anthropic-ai/sdk';
import fs from 'fs';
import path from 'path';

// ---------- Config ----------
const TELEGRAM_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const OPENAI_KEY = process.env.OPENAI_API_KEY; // used for Whisper (speech-to-text) and TTS (text-to-speech)

if (!TELEGRAM_TOKEN || !ANTHROPIC_KEY) {
  console.error('Missing TELEGRAM_BOT_TOKEN or ANTHROPIC_API_KEY in environment.');
  process.exit(1);
}

if (!OPENAI_KEY) {
  console.warn('OPENAI_API_KEY not set — voice messages will not be transcribed.');
}

const bot = new TelegramBot(TELEGRAM_TOKEN, { polling: true });
const anthropic = new Anthropic({ apiKey: ANTHROPIC_KEY });

// ---------- Simple file-based persistence ----------
// Each Telegram user gets their own memory file + conversation history.
const DATA_DIR = path.join(process.cwd(), 'data');
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR);

function userFile(chatId) {
  return path.join(DATA_DIR, `user_${chatId}.json`);
}

function loadUserData(chatId) {
  const file = userFile(chatId);
  if (fs.existsSync(file)) {
    try {
      return JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch (e) {
      console.error('Failed to parse user data, starting fresh:', e);
    }
  }
  return { memories: [], history: [] };
}

function saveUserData(chatId, data) {
  fs.writeFileSync(userFile(chatId), JSON.stringify(data, null, 2));
}

// ---------- Memory tools Claude can call ----------
const tools = [
  {
    name: 'remember_fact',
    description:
      "Save a fact about the user to long-term memory so you can recall it in future conversations — birthdays, work deadlines, plans, preferences, names of people in their life, anything worth remembering long-term. Use this proactively whenever the user mentions something memory-worthy, without waiting to be asked.",
    input_schema: {
      type: 'object',
      properties: {
        text: { type: 'string', description: "The fact to remember, written plainly." },
        category: {
          type: 'string',
          enum: ['date', 'work', 'party', 'person', 'preference', 'other'],
          description: 'One of: date, work, party, person, preference, other',
        },
      },
      required: ['text', 'category'],
    },
  },
  {
    name: 'forget_fact',
    description:
      'Remove a previously remembered fact, e.g. when the user says it is no longer true or relevant.',
    input_schema: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'The text (or close match) of the memory to forget' },
      },
      required: ['text'],
    },
  },
];

function addMemory(userData, text, category) {
  const entry = { text, category: category || 'other', date: new Date().toISOString().slice(0, 10) };
  userData.memories.push(entry);
  return { ok: true, saved: text };
}

function removeMemory(userData, text) {
  const lower = text.trim().toLowerCase();
  const idx = userData.memories.findIndex(
    (m) => m.text.toLowerCase().includes(lower) || lower.includes(m.text.toLowerCase())
  );
  if (idx === -1) return { error: 'Could not find a matching memory to forget.' };
  const removed = userData.memories.splice(idx, 1)[0];
  return { ok: true, removed: removed.text };
}

// ---------- System prompt ----------
function buildSystemPrompt(userData) {
  const base = `You are Jarvis, a sharp, witty personal AI assistant talking to your owner over Telegram, like texting a close friend who's good with tech. Not formal, not robotic, no "as an AI" disclaimers. Keep responses conversational and reasonably short unless they ask for real detail. You have long-term memory: use remember_fact whenever the user mentions something worth keeping track of (dates, work deadlines, plans, people, preferences) — do this proactively, don't wait to be asked. Use forget_fact when something they told you is no longer true. Naturally bring up relevant memories when they fit the conversation.`;

  if (!userData.memories.length) return base;

  const memLines = userData.memories
    .map((m) => `- [${m.category}] ${m.text} (saved ${m.date})`)
    .join('\n');

  return `${base}\n\nHere is what you currently remember about this user:\n${memLines}`;
}

// ---------- Core Claude call with tool loop ----------
async function callClaude(chatId, userText) {
  const userData = loadUserData(chatId);
  userData.history.push({ role: 'user', content: userText });

  // Keep history bounded so requests don't grow unbounded
  const MAX_HISTORY = 40;
  if (userData.history.length > MAX_HISTORY) {
    userData.history = userData.history.slice(-MAX_HISTORY);
  }

  const MAX_TOOL_ITERATIONS = 5;

  for (let i = 0; i < MAX_TOOL_ITERATIONS; i++) {
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 1024,
      system: buildSystemPrompt(userData),
      messages: userData.history,
      tools,
    });

    const toolUseBlocks = response.content.filter((b) => b.type === 'tool_use');

    if (toolUseBlocks.length > 0) {
      userData.history.push({ role: 'assistant', content: response.content });

      const toolResults = [];
      for (const block of toolUseBlocks) {
        let result;
        if (block.name === 'remember_fact') {
          result = addMemory(userData, block.input.text, block.input.category);
        } else if (block.name === 'forget_fact') {
          result = removeMemory(userData, block.input.text);
        } else {
          result = { error: 'Unknown tool' };
        }
        toolResults.push({
          type: 'tool_result',
          tool_use_id: block.id,
          content: JSON.stringify(result),
        });
      }

      userData.history.push({ role: 'user', content: toolResults });
      saveUserData(chatId, userData);
      continue;
    }

    const textBlock = response.content.find((b) => b.type === 'text');
    const reply = textBlock ? textBlock.text : "Sorry, I didn't get a usable reply back.";
    userData.history.push({ role: 'assistant', content: reply });
    saveUserData(chatId, userData);
    return reply;
  }

  saveUserData(chatId, userData);
  return "I got stuck looping on tool calls — try asking again.";
}

// ---------- Voice transcription (OpenAI Whisper) ----------
async function transcribeVoice(fileUrl) {
  if (!OPENAI_KEY) {
    return { error: 'Voice transcription is not configured (missing OPENAI_API_KEY).' };
  }
  try {
    const audioResp = await fetch(fileUrl);
    if (!audioResp.ok) {
      return { error: 'Could not download the voice file from Telegram.' };
    }
    const audioBuffer = await audioResp.arrayBuffer();

    const formData = new FormData();
    formData.append('file', new Blob([audioBuffer], { type: 'audio/ogg' }), 'voice.ogg');
    formData.append('model', 'whisper-1');

    const whisperResp = await fetch('https://api.openai.com/v1/audio/transcriptions', {
      method: 'POST',
      headers: { Authorization: `Bearer ${OPENAI_KEY}` },
      body: formData,
    });

    if (!whisperResp.ok) {
      const errText = await whisperResp.text();
      return { error: 'Whisper API error: ' + errText.slice(0, 200) };
    }

    const data = await whisperResp.json();
    return { ok: true, text: data.text };
  } catch (err) {
    return { error: 'Transcription failed: ' + err.message };
  }
}

// ---------- Text-to-speech (OpenAI TTS) ----------
async function synthesizeSpeech(text) {
  if (!OPENAI_KEY) {
    return { error: 'Voice replies are not configured (missing OPENAI_API_KEY).' };
  }
  try {
    const resp = await fetch('https://api.openai.com/v1/audio/speech', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${OPENAI_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'tts-1',
        voice: 'onyx',
        input: text,
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      return { error: 'TTS API error: ' + errText.slice(0, 200) };
    }

    const buffer = Buffer.from(await resp.arrayBuffer());
    return { ok: true, buffer };
  } catch (err) {
    return { error: 'Speech synthesis failed: ' + err.message };
  }
}

// ---------- Telegram wiring ----------
bot.on('message', async (msg) => {
  const chatId = msg.chat.id;
  let text = msg.text;
  let wasVoice = false;

  if (msg.voice) {
    wasVoice = true;
    bot.sendChatAction(chatId, 'typing');
    try {
      const fileLink = await bot.getFileLink(msg.voice.file_id);
      const transcription = await transcribeVoice(fileLink);
      if (transcription.error) {
        bot.sendMessage(chatId, "Couldn't understand that voice message: " + transcription.error);
        return;
      }
      text = transcription.text;
      if (!text || !text.trim()) {
        bot.sendMessage(chatId, "Got the voice message but couldn't make out any words — try again?");
        return;
      }
    } catch (err) {
      console.error('Voice handling error:', err);
      bot.sendMessage(chatId, "Hit an error processing that voice message: " + err.message);
      return;
    }
  }

  if (!text) {
    bot.sendMessage(chatId, "I can only handle text or voice messages right now.");
    return;
  }

  if (text === '/start') {
    bot.sendMessage(
      chatId,
      "Hey, I'm Jarvis. Just talk to me normally — I'll remember things you tell me and bring them up later. What's up?"
    );
    return;
  }

  if (text === '/forget_everything') {
    saveUserData(chatId, { memories: [], history: [] });
    bot.sendMessage(chatId, "Done — wiped everything I remembered about you.");
    return;
  }

  bot.sendChatAction(chatId, wasVoice ? 'record_voice' : 'typing');

  try {
    const reply = await callClaude(chatId, text);

    if (wasVoice) {
      const speech = await synthesizeSpeech(reply);
      if (speech.ok) {
        bot.sendVoice(chatId, speech.buffer, {}, { filename: 'reply.mp3', contentType: 'audio/mpeg' });
      } else {
        // Fall back to text if TTS fails, so the user still gets the answer.
        bot.sendMessage(chatId, reply);
      }
    } else {
      bot.sendMessage(chatId, reply);
    }
  } catch (err) {
    console.error('Error handling message:', err);
    bot.sendMessage(chatId, "Hit an error on my end: " + err.message);
  }
});

bot.on('polling_error', (err) => {
  console.error('Polling error:', err.message);
});

console.log('Jarvis Telegram bot is running...');
