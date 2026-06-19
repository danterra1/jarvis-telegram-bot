import 'dotenv/config';
import TelegramBot from 'node-telegram-bot-api';
import Anthropic from '@anthropic-ai/sdk';
import fs from 'fs';
import path from 'path';

// ---------- Config ----------
const TELEGRAM_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const OPENAI_KEY = process.env.OPENAI_API_KEY; // used for Whisper (speech-to-text) and TTS (text-to-speech)
const TAVILY_KEY = process.env.TAVILY_API_KEY; // used for web search

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
  return { memories: [], history: [], reminders: [] };
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
          enum: ['date', 'work', 'party', 'person', 'preference', 'location', 'other'],
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
  {
    name: 'web_search',
    description:
      'Search the web for current information, facts, news, recommendations, or anything you are not sure about or that needs up-to-date info. Use this whenever the user asks about current events, wants suggestions/recommendations, or asks something you cannot confidently answer from memory alone.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'The search query' },
      },
      required: ['query'],
    },
  },
  {
    name: 'request_location',
    description:
      "Ask the user to share their current location, e.g. when they ask for something nearby (restaurants, weather, directions) and you don't already know where they are. This sends them a one-tap button to share their location — it does not return the location immediately, the user has to tap it. After calling this, tell the user you've sent a button for them to tap.",
    input_schema: {
      type: 'object',
      properties: {
        reason: { type: 'string', description: 'Brief reason why location is needed, e.g. "find restaurants nearby"' },
      },
      required: ['reason'],
    },
  },
  {
    name: 'set_reminder',
    description:
      "Schedule a reminder that will be proactively sent to the user at a specific future time, even if they haven't messaged you. Use this whenever the user asks to be reminded or pinged about something at a specific time or after a delay (e.g. 'remind me in 20 minutes', 'ping me tomorrow at 9am about the meeting'). You must convert their request into an exact ISO 8601 datetime using the current date/time and their timezone if known.",
    input_schema: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'What to remind the user about' },
        when_iso: { type: 'string', description: 'Exact ISO 8601 datetime to send the reminder, e.g. 2026-06-19T14:30:00Z' },
      },
      required: ['text', 'when_iso'],
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

function setReminder(userData, text, whenIso) {
  const when = new Date(whenIso);
  if (isNaN(when.getTime())) {
    return { error: 'Invalid date/time format for reminder.' };
  }
  if (!userData.reminders) userData.reminders = [];
  userData.reminders.push({ text, when: when.toISOString(), id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}` });
  return { ok: true, scheduled: text, when: when.toISOString() };
}

// ---------- Web search (Tavily) ----------
async function webSearch(query) {
  if (!TAVILY_KEY) {
    return { error: 'Web search is not configured (missing TAVILY_API_KEY).' };
  }
  try {
    const resp = await fetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${TAVILY_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, max_results: 5, include_answer: true }),
    });

    if (!resp.ok) {
      return { error: 'Search API error ' + resp.status };
    }

    const data = await resp.json();
    const results = (data.results || []).slice(0, 5).map((r) => ({
      title: r.title,
      url: r.url,
      snippet: r.content,
    }));
    const answerBox = data.answer ? { answer: data.answer } : null;
    return { query, answerBox, results };
  } catch (err) {
    return { error: 'Search failed: ' + err.message };
  }
}

// ---------- Location sharing ----------
async function sendLocationRequest(chatId, reason, alsoSpeak) {
  const promptText = reason
    ? `I need your location to ${reason}. Tap the button below to share it:`
    : 'I need your location. Tap the button below to share it:';
  await bot.sendMessage(chatId, promptText, {
    reply_markup: {
      keyboard: [[{ text: '📍 Share my location', request_location: true }]],
      one_time_keyboard: true,
      resize_keyboard: true,
    },
  });
  if (alsoSpeak) {
    const speech = await synthesizeSpeech(
      reason ? `I need your location to ${reason}. I've sent you a button to tap.` : "I need your location. I've sent you a button to tap."
    );
    if (speech.ok) {
      bot.sendVoice(chatId, speech.buffer, {}, { filename: 'reply.mp3', contentType: 'audio/mpeg' });
    }
  }
}

async function reverseGeocode(lat, lon) {
  try {
    const resp = await fetch(
      `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=10`,
      { headers: { 'User-Agent': 'jarvis-telegram-bot/1.0' } }
    );
    if (!resp.ok) return null;
    const data = await resp.json();
    const addr = data.address || {};
    const place = addr.city || addr.town || addr.village || addr.county || data.display_name;
    return place || null;
  } catch (err) {
    return null;
  }
}

// ---------- System prompt ----------
function buildSystemPrompt(userData) {
  const nowIso = new Date().toISOString();
  const base = `You are Jarvis, a sharp, witty personal AI assistant talking to your owner over Telegram, like texting a close friend who's good with tech. Not formal, not robotic, no "as an AI" disclaimers. Keep responses conversational and reasonably short unless they ask for real detail. The current UTC date/time is ${nowIso} — use this as "now" when the user gives a relative time (e.g. "in 20 minutes", "tomorrow at 9am"). You have long-term memory: use remember_fact whenever the user mentions something worth keeping track of (dates, work deadlines, plans, people, preferences) — do this proactively, don't wait to be asked. Use forget_fact when something they told you is no longer true. Naturally bring up relevant memories when they fit the conversation. You also have web_search — use it whenever the user asks for recommendations, current events, or anything you're not confidently sure of. When you get search results back, describe what you found in your own words, but DO include the actual name, address/link, and key details for anything the user would want to visit, click, or act on (restaurants, businesses, articles, products) — don't strip out useful URLs or names just to paraphrase, the user needs that information to actually use it. If the user asks for something location-dependent (nearby restaurants, local weather, directions) and you don't already have a saved location for them in memory, use request_location to ask them to share it — then wait for them to tap the button before searching. Use set_reminder whenever the user asks to be reminded or pinged about something at a future time — convert their relative time into an exact ISO datetime using the current time above, and account for their saved location's timezone if you can infer it.`;

  if (!userData.memories.length) return base;

  const memLines = userData.memories
    .map((m) => `- [${m.category}] ${m.text} (saved ${m.date})`)
    .join('\n');

  let result = `${base}\n\nHere is what you currently remember about this user:\n${memLines}`;

  if (userData.reminders && userData.reminders.length) {
    const remLines = userData.reminders
      .map((r) => `- "${r.text}" scheduled for ${r.when}`)
      .join('\n');
    result += `\n\nUpcoming reminders already scheduled for this user:\n${remLines}`;
  }

  return result;
}

// ---------- Core Claude call with tool loop ----------
async function callClaude(chatId, userText, wasVoice) {
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
        } else if (block.name === 'web_search') {
          result = await webSearch(block.input.query);
        } else if (block.name === 'request_location') {
          await sendLocationRequest(chatId, block.input.reason, wasVoice);
          userData.pendingWasVoice = wasVoice;
          saveUserData(chatId, userData);
          result = { ok: true, note: 'Location request button sent to the user. Tell them you sent it and to tap it.' };
        } else if (block.name === 'set_reminder') {
          result = setReminder(userData, block.input.text, block.input.when_iso);
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

  if (msg.location) {
    const { latitude, longitude } = msg.location;
    bot.sendChatAction(chatId, 'typing');
    const place = await reverseGeocode(latitude, longitude);
    const userData = loadUserData(chatId);
    const locationText = place
      ? `User's current location is ${place} (lat ${latitude}, lon ${longitude})`
      : `User's current location is lat ${latitude}, lon ${longitude}`;
    // Replace any previous location memory rather than stacking duplicates
    userData.memories = userData.memories.filter((m) => m.category !== 'location');
    addMemory(userData, locationText, 'location');
    saveUserData(chatId, userData);
    bot.sendMessage(chatId, place ? `Got it — I'll remember you're around ${place}.` : 'Got your location, thanks.', {
      reply_markup: { remove_keyboard: true },
    });
    // Feed the location into the conversation so Claude can pick back up
    // whatever it was trying to do before it asked for the location.
    const wasVoiceBeforeLocation = userData.pendingWasVoice || false;
    try {
      const reply = await callClaude(
        chatId,
        `[The user just shared their location: ${locationText}. Continue helping with whatever you needed the location for.]`,
        wasVoiceBeforeLocation
      );
      if (reply) {
        if (wasVoiceBeforeLocation) {
          const speech = await synthesizeSpeech(reply);
          if (speech.ok) {
            bot.sendVoice(chatId, speech.buffer, {}, { filename: 'reply.mp3', contentType: 'audio/mpeg' });
          } else {
            bot.sendMessage(chatId, reply);
          }
        } else {
          bot.sendMessage(chatId, reply);
        }
      }
    } catch (err) {
      console.error('Error continuing after location share:', err);
    }
    return;
  }

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
    sendLocationRequest(chatId, 'know your timezone and location for things like reminders, nearby suggestions, and local weather', false);
    return;
  }

  if (text === '/forget_everything') {
    saveUserData(chatId, { memories: [], history: [] });
    bot.sendMessage(chatId, "Done — wiped everything I remembered about you.");
    return;
  }

  bot.sendChatAction(chatId, wasVoice ? 'record_voice' : 'typing');

  try {
    const reply = await callClaude(chatId, text, wasVoice);

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

// ---------- Reminder scheduler ----------
// Checks every minute for any due reminders across all users and proactively
// pings them, even if they haven't sent a message recently.
function checkReminders() {
  let files;
  try {
    files = fs.readdirSync(DATA_DIR).filter((f) => f.startsWith('user_') && f.endsWith('.json'));
  } catch (err) {
    return;
  }

  const now = Date.now();

  for (const file of files) {
    const chatId = file.slice('user_'.length, -'.json'.length);
    let userData;
    try {
      userData = JSON.parse(fs.readFileSync(path.join(DATA_DIR, file), 'utf8'));
    } catch (err) {
      continue;
    }
    if (!userData.reminders || !userData.reminders.length) continue;

    const due = userData.reminders.filter((r) => new Date(r.when).getTime() <= now);
    if (!due.length) continue;

    userData.reminders = userData.reminders.filter((r) => new Date(r.when).getTime() > now);
    saveUserData(chatId, userData);

    for (const reminder of due) {
      bot.sendMessage(chatId, `⏰ Reminder: ${reminder.text}`).catch((err) => {
        console.error('Failed to send reminder to', chatId, err.message);
      });
    }
  }
}

setInterval(checkReminders, 60 * 1000);

console.log('Jarvis Telegram bot is running...');
