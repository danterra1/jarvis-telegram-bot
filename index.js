import 'dotenv/config';
import TelegramBot from 'node-telegram-bot-api';
import Anthropic from '@anthropic-ai/sdk';
import fs from 'fs';
import path from 'path';
import http from 'http';

// ---------- Config ----------
const TELEGRAM_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const OPENAI_KEY = process.env.OPENAI_API_KEY; // used for Whisper (speech-to-text) and TTS (text-to-speech)
const TAVILY_KEY = process.env.TAVILY_API_KEY; // used for web search
const VAPI_KEY = process.env.VAPI_API_KEY; // used for outbound phone calls (restaurant bookings, etc.)
const VAPI_PHONE_NUMBER_ID = process.env.VAPI_PHONE_NUMBER_ID; // the Vapi phone number to call FROM
const PUBLIC_URL = process.env.PUBLIC_URL; // public base URL of this service, used for Vapi's webhook callback

if (!TELEGRAM_TOKEN || !ANTHROPIC_KEY) {
  console.error('Missing TELEGRAM_BOT_TOKEN or ANTHROPIC_API_KEY in environment.');
  process.exit(1);
}

if (!OPENAI_KEY) {
  console.warn('OPENAI_API_KEY not set — voice messages will not be transcribed.');
}

if (!VAPI_KEY) {
  console.warn('VAPI_API_KEY not set — outbound booking calls will not work.');
}

const bot = new TelegramBot(TELEGRAM_TOKEN, { polling: true });
const anthropic = new Anthropic({ apiKey: ANTHROPIC_KEY });

// ---------- Simple file-based persistence ----------
// Each Telegram user gets their own memory file + conversation history.
// This file is the durable "long-term memory" store for that user — it is
// the source of truth Claude can always fall back on, independent of
// whatever happens to be loaded into its context window on a given turn.
const DATA_DIR = path.join(process.cwd(), 'data');
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR);

function userFile(chatId) {
  return path.join(DATA_DIR, `user_${chatId}.json`);
}

// Bump this if the on-disk shape ever changes incompatibly.
const DATA_VERSION = 2;

function emptyUserData(username, firstName) {
  return { version: DATA_VERSION, memories: [], history: [], reminders: [], username: username || '', firstName: firstName || '' };
}

function loadUserData(chatId) {
  const file = userFile(chatId);
  if (fs.existsSync(file)) {
    try {
      const data = JSON.parse(fs.readFileSync(file, 'utf8'));
      // Backfill defaults for older files / partial writes so the rest of
      // the code never has to null-check these.
      if (!Array.isArray(data.memories)) data.memories = [];
      if (typeof data.username === 'undefined') data.username = '';
      if (typeof data.firstName === 'undefined') data.firstName = '';
      if (!Array.isArray(data.history)) data.history = [];
      if (!Array.isArray(data.reminders)) data.reminders = [];
      data.version = DATA_VERSION;
      return data;
    } catch (e) {
      // Corrupt file — don't silently wipe the user's memory. Keep the
      // broken file around for inspection/recovery and start a fresh
      // in-memory object for this turn only; it will be saved fresh on
      // the next successful write, but we log loudly so this is visible.
      console.error(`Failed to parse user data for ${chatId}, file may be corrupt:`, e);
      try {
        fs.copyFileSync(file, file + '.corrupt-' + Date.now());
      } catch (copyErr) {
        console.error('Could not back up corrupt user file:', copyErr);
      }
      return emptyUserData();
    }
  }
  return emptyUserData();
}

// Atomic-ish write: write to a temp file then rename, so a crash mid-write
// never leaves a half-written / corrupt JSON file on disk.
function saveUserData(chatId, data) {
  const file = userFile(chatId);
  const tmp = file + '.tmp';
  try {
    fs.writeFileSync(tmp, JSON.stringify(data, null, 2));
    fs.renameSync(tmp, file);
  } catch (err) {
    console.error(`Failed to save user data for ${chatId}:`, err);
  }
}

// ---------- Memory helpers ----------
// Every memory has: id, text, category, date (created), updatedAt.
// IDs let forget/update target an exact memory instead of fuzzy substring
// matching against text, which is fragile and can hit the wrong entry.
function makeMemoryId() {
  return `m_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function normalize(s) {
  return (s || '').toLowerCase().trim();
}

// Very lightweight similarity check used to detect "this is probably the
// same fact as one we already have" so remember_fact updates in place
// instead of creating duplicates. Token-overlap based, no extra deps.
function tokenOverlapScore(a, b) {
  const ta = new Set(normalize(a).split(/\W+/).filter((w) => w.length > 2));
  const tb = new Set(normalize(b).split(/\W+/).filter((w) => w.length > 2));
  if (!ta.size || !tb.size) return 0;
  let shared = 0;
  for (const w of ta) if (tb.has(w)) shared++;
  return shared / Math.max(ta.size, tb.size);
}

function findClosestMemory(userData, text, category) {
  let best = null;
  let bestScore = 0;
  for (const m of userData.memories) {
    if (category && m.category !== category) continue;
    const score = tokenOverlapScore(m.text, text);
    if (score > bestScore) {
      bestScore = score;
      best = m;
    }
  }
  return bestScore >= 0.5 ? best : null;
}

function addMemory(userData, text, category) {
  const existing = findClosestMemory(userData, text, category);
  const today = new Date().toISOString().slice(0, 10);
  if (existing) {
    existing.text = text;
    existing.updatedAt = today;
    return { ok: true, updated: text, id: existing.id, note: 'Updated an existing similar memory instead of duplicating it.' };
  }
  const entry = { id: makeMemoryId(), text, category: category || 'other', date: today, updatedAt: today };
  userData.memories.push(entry);
  return { ok: true, saved: text, id: entry.id };
}

function removeMemory(userData, text) {
  if (!userData.memories.length) return { error: 'No memories stored for this user.' };

  // Prefer an exact id match if the caller happens to pass one.
  let idx = userData.memories.findIndex((m) => m.id === text);

  // Then try substring match either direction (legacy behavior).
  if (idx === -1) {
    const lower = normalize(text);
    idx = userData.memories.findIndex(
      (m) => normalize(m.text).includes(lower) || lower.includes(normalize(m.text))
    );
  }

  // Fall back to token-overlap similarity so close paraphrases still match.
  if (idx === -1) {
    const best = findClosestMemory(userData, text, null);
    if (best) idx = userData.memories.findIndex((m) => m.id === best.id);
  }

  if (idx === -1) {
    return { error: 'Could not find a matching memory to forget.' };
  }
  const removed = userData.memories.splice(idx, 1)[0];
  return { ok: true, removed: removed.text };
}

// Active recall: lets Claude explicitly search the durable memory store on
// disk by keyword/category instead of relying solely on whatever subset got
// pre-loaded into the system prompt this turn. This is the "if I forget
// something, go check storage" path.
function recallMemory(userData, query, category) {
  const q = normalize(query);
  const tokens = q.split(/\W+/).filter((w) => w.length > 1);

  const scored = userData.memories
    .filter((m) => !category || m.category === category)
    .map((m) => {
      const text = normalize(m.text);
      let score = 0;
      if (q && text.includes(q)) score += 2; // direct substring is a strong signal
      for (const t of tokens) if (text.includes(t)) score += 0.5;
      return { m, score };
    })
    .filter((s) => s.score > 0 || !q) // empty query = return everything (e.g. for a category browse)
    .sort((a, b) => b.score - a.score)
    .slice(0, 15)
    .map((s) => s.m);

  if (!scored.length) {
    return {
      ok: true,
      found: false,
      results: [],
      note: query
        ? `No stored memory matched "${query}". This does not mean it was never saved — try a different keyword, or tell the user you don't have that on record.`
        : 'No memories stored for this user yet.',
    };
  }

  return {
    ok: true,
    found: true,
    results: scored.map((m) => ({ id: m.id, text: m.text, category: m.category, date: m.date, updatedAt: m.updatedAt })),
  };
}

function setReminder(userData, text, whenIso, recurrence, isGiftOccasion) {
  const when = new Date(whenIso);
  if (isNaN(when.getTime())) {
    return { error: 'Invalid date/time format for reminder.' };
  }
  if (!userData.reminders) userData.reminders = [];
  userData.reminders.push({
    text,
    when: when.toISOString(),
    recurrence: recurrence || 'none',
    isGiftOccasion: !!isGiftOccasion,
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  });
  return { ok: true, scheduled: text, when: when.toISOString(), recurrence: recurrence || 'none' };
}

// ---------- Memory tools Claude can call ----------
const tools = [
  {
    name: 'remember_fact',
    description:
      "Save a fact about the user to long-term memory so you can recall it in future conversations — birthdays, work deadlines, plans, preferences, names of people in their life, anything worth remembering long-term. Use this proactively whenever the user mentions something memory-worthy, without waiting to be asked. If a very similar memory already exists, this will update it in place instead of creating a duplicate, so it's safe to call even when correcting or refining something you already knew.",
    input_schema: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'The fact to remember, written plainly.' },
        category: {
          type: 'string',
          enum: ['date', 'work', 'party', 'person', 'preference', 'location', 'other'],
          description: 'One of: date, work, party, person, preference, location, other',
        },
      },
      required: ['text', 'category'],
    },
  },
  {
    name: 'recall_memory',
    description:
      "Actively search the user's full long-term memory store on disk for anything matching a keyword or topic. Use this whenever you're not sure whether you already know something about the user, when a memory you'd expect to have isn't showing up in what's already loaded into this conversation, or before saying 'I don't know that about you' — check storage first. You can pass an empty query with a category to browse everything in that category (e.g. all 'person' memories). This searches the same durable store remember_fact writes to, so it will find things even if they aren't currently visible in your context.",
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Keyword or phrase to search for. Leave empty to browse by category instead.' },
        category: {
          type: 'string',
          enum: ['date', 'work', 'party', 'person', 'preference', 'location', 'other'],
          description: 'Optional: restrict the search to one category.',
        },
      },
      required: [],
    },
  },
  {
    name: 'forget_fact',
    description:
      'Remove a previously remembered fact, e.g. when the user says it is no longer true or relevant. Pass the memory id if you have it from a recall_memory or remember_fact result for an exact match, otherwise pass the text and it will be matched fuzzily.',
    input_schema: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'The memory id (preferred if known) or the text (close match) of the memory to forget' },
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
      "Schedule a reminder that will be proactively sent to the user at a specific future time, even if they haven't messaged you. Use this whenever the user asks to be reminded or pinged about something at a specific time or after a delay (e.g. 'remind me in 20 minutes', 'ping me tomorrow at 9am about the meeting'), or for recurring things like birthdays or anniversaries. You must convert their request into an exact ISO 8601 datetime using the current date/time and their timezone if known. If the thing being scheduled is naturally yearly (birthday, anniversary) ask the user if they want it to repeat every year, and set recurrence accordingly.",
    input_schema: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'What to remind the user about' },
        when_iso: { type: 'string', description: 'Exact ISO 8601 datetime to send the reminder, e.g. 2026-06-19T14:30:00Z' },
        recurrence: {
          type: 'string',
          enum: ['none', 'yearly', 'monthly', 'weekly'],
          description: 'How often this reminder repeats. Use "yearly" for birthdays/anniversaries, "none" for one-off reminders.',
        },
        is_gift_occasion: {
          type: 'boolean',
          description: 'True if this is a birthday, anniversary, or other occasion where the user might want gift suggestions when the reminder fires.',
        },
      },
      required: ['text', 'when_iso'],
    },
  },
  {
    name: 'make_booking_call',
    description:
      "Place a real outbound phone call on the user's behalf to book a restaurant reservation (or similar booking). Use this when the user asks you to call and book a table somewhere. You need a phone number for the business — if you don't already have one in memory or from a recent web_search, call web_search first to try to find it. If you still can't find a reliable phone number, do NOT guess one — ask the user to provide it instead. Before calling this tool, briefly tell the user what you're about to do (who you're calling, for what, at what time) in your reply text, since the call happens immediately and asynchronously. The outcome of the call will be reported back to the user separately once the call finishes, it is not returned by this tool.",
    input_schema: {
      type: 'object',
      properties: {
        business_name: { type: 'string', description: 'Name of the restaurant or business being called' },
        phone_number: { type: 'string', description: 'Phone number to call, in E.164 format if possible (e.g. +15551234567)' },
        party_size: { type: 'integer', description: 'Number of people for the reservation' },
        date_time_description: { type: 'string', description: 'Human-readable date and time for the reservation, e.g. "tonight at 8pm" or "Saturday June 21st at 7:30pm"' },
        reservation_name: { type: 'string', description: "Name to put the reservation under (usually the user's name)" },
        special_requests: { type: 'string', description: 'Any special requests, e.g. "window seat", "highchair needed", dietary notes. Optional.' },
      },
      required: ['business_name', 'phone_number', 'party_size', 'date_time_description', 'reservation_name'],
    },
  },
];

// ---------- URL validation helper ----------
// Only accept well-formed http(s) URLs. Anything else (empty, malformed,
// relative, javascript:, etc.) is dropped so we never hand Claude — or the
// user — a broken / fake link.
function isWellFormedUrl(u) {
  if (!u || typeof u !== 'string') return false;
  try {
    const parsed = new URL(u.trim());
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch (err) {
    return false;
  }
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
    const results = (data.results || [])
      .slice(0, 5)
      .map((r) => ({
        title: r.title,
        url: isWellFormedUrl(r.url) ? r.url.trim() : null,
        snippet: r.content,
      }));
    const answerBox = data.answer ? { answer: data.answer } : null;
    return { query, answerBox, results };
  } catch (err) {
    return { error: 'Search failed: ' + err.message };
  }
}

// ---------- Extract URLs from a reply (for voice+text dual send) ----------
function extractUrls(text) {
  if (!text) return [];
  const matches = text.match(/https?:\/\/[^\s)\]]+/g) || [];
  const cleaned = matches
    .map((u) => u.replace(/[.,;:!?'")\]]+$/, '')) // strip trailing punctuation
    .filter(isWellFormedUrl);
  return [...new Set(cleaned)];
}

// ---------- Outbound booking calls (Vapi) ----------
// Tracks in-flight Vapi calls so the webhook callback knows which Telegram
// chat to report the outcome back to. Kept in memory + persisted to disk so
// a redeploy mid-call doesn't lose the mapping.
const CALLS_FILE = path.join(DATA_DIR, 'active_calls.json');

function loadActiveCalls() {
  if (fs.existsSync(CALLS_FILE)) {
    try {
      return JSON.parse(fs.readFileSync(CALLS_FILE, 'utf8'));
    } catch (e) {
      return {};
    }
  }
  return {};
}

function saveActiveCalls(calls) {
  const tmp = CALLS_FILE + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(calls, null, 2));
  fs.renameSync(tmp, CALLS_FILE);
}

function buildBookingSystemPrompt({ business_name, party_size, date_time_description, reservation_name, special_requests }) {
  return `You are a personal assistant calling ${business_name} on behalf of ${reservation_name} to book a restaurant reservation. Politely ask for a table for ${party_size} people, for ${date_time_description}, under the name ${reservation_name}.${special_requests ? ` Special request to mention: ${special_requests}.` : ''} If that exact time isn't available, ask what times are available close to it, note the closest alternative, and do NOT confirm an alternate time yourself — just report what was offered. If anyone asks whether they are speaking with a real person or an AI, be honest and say you are an AI assistant calling on behalf of ${reservation_name}. Be brief and polite, and end the call once the booking is confirmed, declined, or you have an alternative time to report back.`;
}

async function makeBookingCall(chatId, args) {
  if (!VAPI_KEY) {
    return { error: 'Outbound calling is not configured (missing VAPI_API_KEY).' };
  }
  if (!VAPI_PHONE_NUMBER_ID) {
    return { error: 'Outbound calling is not fully configured (missing VAPI_PHONE_NUMBER_ID).' };
  }
  if (!PUBLIC_URL) {
    return { error: 'Outbound calling is not fully configured (missing PUBLIC_URL for the callback webhook).' };
  }

  const systemPrompt = buildBookingSystemPrompt(args);

  try {
    const resp = await fetch('https://api.vapi.ai/call', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${VAPI_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        phoneNumberId: VAPI_PHONE_NUMBER_ID,
        customer: { number: args.phone_number },
        assistant: {
          firstMessage: `Hi, I'm calling to book a table for ${args.party_size}, for ${args.date_time_description}.`,
          model: {
            provider: 'anthropic',
            model: 'claude-sonnet-4-6',
            messages: [{ role: 'system', content: systemPrompt }],
          },
          voice: { provider: 'playht', voiceId: 'jennifer' },
          serverUrl: `${PUBLIC_URL}/vapi-webhook`,
        },
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      return { error: 'Vapi API error: ' + errText.slice(0, 200) };
    }

    const data = await resp.json();
    if (!data.id) {
      return { error: 'Vapi did not return a call id: ' + JSON.stringify(data).slice(0, 200) };
    }

    const calls = loadActiveCalls();
    calls[data.id] = {
      chatId: String(chatId),
      business_name: args.business_name,
      date_time_description: args.date_time_description,
      startedAt: new Date().toISOString(),
    };
    saveActiveCalls(calls);

    return { ok: true, call_id: data.id, note: `Call placed to ${args.business_name}. The outcome will be reported back once the call ends.` };
  } catch (err) {
    return { error: 'Failed to place call: ' + err.message };
  }
}

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
// Only a bounded, recent slice of memories is inlined directly into the
// prompt every turn (cheap "working memory"). Everything else still lives
// in the user's data file and is reachable via recall_memory — so the
// store Claude can fall back on is never smaller than what's on disk, even
// once the person has hundreds of memories.
const INLINE_MEMORY_LIMIT = 25;


function buildSystemPrompt(userData) {
  const nowIso = new Date().toISOString();
  const base = `You are Jarvis, a sharp, witty personal AI assistant talking to your owner over Telegram, like texting a close friend who's good with tech. Not formal, not robotic, no "as an AI" disclaimers. Keep responses conversational and reasonably short unless they ask for real detail. The current UTC date/time is ${nowIso} — use this as "now" when the user gives a relative time (e.g. "in 20 minutes", "tomorrow at 9am"). 

You have a durable long-term memory store for this user, saved to disk, that persists across all conversations. Use remember_fact proactively whenever the user mentions something worth keeping track of (dates, work deadlines, plans, people, preferences, locations) — don't wait to be asked. Use forget_fact when something they told you is no longer true, and remember_fact again with corrected info if they're updating rather than removing a fact. Below you'll see a list of recent/frequent memories already loaded for convenience, but that list is NOT your full memory — it's just a quick-access subset. If you're asked about something not in that list, if you're unsure whether you already know something, or if you ever catch yourself about to say "I don't know that about you," call recall_memory first to actively search the full store before answering — don't assume something was never saved just because it isn't sitting in front of you right now. This matters: the user is relying on you to actually have persistent memory, not to fake it from a short list.

You also have web_search — use it whenever the user asks for recommendations, current events, or anything you're not confidently sure of. When you get search results back, describe what you found in your own words, but DO include the actual name, address/link, and key details for anything the user would want to visit, click, or act on (restaurants, businesses, articles, products) — don't strip out useful URLs or names just to paraphrase, the user needs that information to actually use it. Each search result's "url" field will be null if no reliable link was found for that result — if a result has a null url, mention the name/snippet but do NOT invent, guess, or fabricate a URL for it; just skip the link for that one item. Never present a link you are not certain came directly from the search results. If the user asks for something location-dependent (nearby restaurants, local weather, directions) and you don't already have a saved location for them in memory, use request_location to ask them to share it — then wait for them to tap the button before searching. Use set_reminder whenever the user asks to be reminded or pinged about something at a future time — convert their relative time into an exact ISO datetime using the current time above, and account for their saved location's timezone if you can infer it. Be a genuinely proactive personal assistant: when a request is ambiguous or underspecified, ask a short clarifying question instead of guessing — e.g. if asked to "remind me about birthdays" without specifics, ask whose birthday, what date, and whether it repeats every year. For birthdays, anniversaries, or other yearly occasions, set recurrence to "yearly" and set is_gift_occasion to true so you can proactively suggest gift ideas when the reminder fires. You also have make_booking_call — use it when the user asks you to call and book a restaurant reservation on their behalf. Find the business's phone number via web_search if you don't already have it; never invent a phone number, ask the user for it if search doesn't find a reliable one. Before calling this tool, tell the user in your reply exactly what you're about to do (who you're calling, party size, and time) since the call happens immediately. The result of the call is reported back separately once it finishes, not in this turn. Treat every conversation as a chance to learn more about the owner — their relationships, preferences, routines — and use that to give sharper, more personalized help over time.`;

  if (!userData.memories.length) return base;

  // Most-recently-updated memories first, capped, so the prompt doesn't grow
  // without bound as a user's memory store grows over months of use.
  const sorted = [...userData.memories].sort((a, b) => (b.updatedAt || b.date).localeCompare(a.updatedAt || a.date));
  const shown = sorted.slice(0, INLINE_MEMORY_LIMIT);
  const omittedCount = userData.memories.length - shown.length;

  const memLines = shown
    .map((m) => `- [${m.category}] ${m.text} (id: ${m.id}, saved ${m.date}${m.updatedAt && m.updatedAt !== m.date ? `, updated ${m.updatedAt}` : ''})`)
    .join('\n');

  let result = `${base}\n\nQuick-access memory (${shown.length} of ${userData.memories.length} total memories — most recently updated first):\n${memLines}`;
  if (omittedCount > 0) {
    result += `\n\n...and ${omittedCount} more memories not shown here. Use recall_memory to search them if relevant.`;
  }

  if (userData.reminders && userData.reminders.length) {
    const remLines = userData.reminders
      .map((r) => `- "${r.text}" scheduled for ${r.when}${r.recurrence && r.recurrence !== 'none' ? ` (repeats ${r.recurrence})` : ''}`)
      .join('\n');
    result += `\n\nUpcoming reminders already scheduled for this user:\n${remLines}`;
  }

  return result;
}

// ---------- Core Claude call with tool loop ----------
async function callClaude(chatId, userText, wasVoice, username) {
  const userData = loadUserData(chatId);
  // Always keep the stored identity up-to-date
  if (username && !userData.username) { userData.username = username; saveUserData(chatId, userData); }
  userData.history.push({ role: 'user', content: userText });

  // Keep history bounded so requests don't grow unbounded
  const MAX_HISTORY = 20;
  if (userData.history.length > MAX_HISTORY) {
    userData.history = userData.history.slice(-MAX_HISTORY);
  }

  const MAX_TOOL_ITERATIONS = 5;
  // Collects every web_search result used during this turn. If any search
  // ran, we always send a text companion with the names/links/addresses —
  // independent of whether the reply text happens to contain a raw URL,
  // since Claude often paraphrases ("Mario's at 123 Main St") without
  // typing out a literal link.
  const searchResultsThisTurn = [];
  let totalAnthropicTokens = 0;
  let totalEventCounts = {};

  for (let i = 0; i < MAX_TOOL_ITERATIONS; i++) {
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 600,
      system: buildSystemPrompt(userData),
      messages: userData.history,
      tools,
    });

    totalAnthropicTokens += (response.usage?.input_tokens || 0) + (response.usage?.output_tokens || 0);
    const toolUseBlocks = response.content.filter((b) => b.type === 'tool_use');

    if (toolUseBlocks.length > 0) {
      userData.history.push({ role: 'assistant', content: response.content });

      const toolResults = [];
      for (const block of toolUseBlocks) {
        let result;
        if (block.name === 'remember_fact') {
          result = addMemory(userData, block.input.text, block.input.category);
          saveUserData(chatId, userData); // persist immediately, don't wait for end of turn
        } else if (block.name === 'recall_memory') {
          result = recallMemory(userData, block.input.query || '', block.input.category);
        } else if (block.name === 'forget_fact') {
          result = removeMemory(userData, block.input.text);
          saveUserData(chatId, userData);
        } else if (block.name === 'web_search') {
          result = await webSearch(block.input.query);
          if (result && !result.error && result.results) {
            searchResultsThisTurn.push(...result.results);
            if (!totalEventCounts) totalEventCounts = {}; totalEventCounts.searches = (totalEventCounts.searches || 0) + 1;
          }
        } else if (block.name === 'request_location') {
          await sendLocationRequest(chatId, block.input.reason, wasVoice);
          userData.pendingWasVoice = wasVoice;
          saveUserData(chatId, userData);
          result = { ok: true, note: 'Location request button sent to the user. Tell them you sent it and to tap it.' };
        } else if (block.name === 'set_reminder') {
          result = setReminder(userData, block.input.text, block.input.when_iso, block.input.recurrence, block.input.is_gift_occasion);
          totalEventCounts.reminders = (totalEventCounts.reminders || 0) + 1;
          saveUserData(chatId, userData);
        } else if (block.name === 'make_booking_call') {
          result = await makeBookingCall(chatId, block.input);
          totalEventCounts.calls = (totalEventCounts.calls || 0) + 1;
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
    return { reply, searchResults: searchResultsThisTurn, anthropicTokens: totalAnthropicTokens, eventCounts: totalEventCounts };
  }

  saveUserData(chatId, userData);
  return { reply: "I got stuck looping on tool calls — try asking again.", searchResults: searchResultsThisTurn, anthropicTokens: totalAnthropicTokens, eventCounts: totalEventCounts };
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

// ---------- Send a reply, respecting voice-in/voice-out, with a text
// ---------- companion message whenever there's searched info or links to share ----------
async function sendReply(chatId, reply, wasVoice, searchResults) {
  const urls = extractUrls(reply);
  const namedResults = (searchResults || []).filter((r) => r && r.title);

  if (!wasVoice) {
    // Text in -> text out, always. Links/details are already inline in the text.
    await bot.sendMessage(chatId, reply);
    return;
  }

  // Voice in -> voice out.
  const speech = await synthesizeSpeech(reply);
  if (speech.ok) {
    await bot.sendVoice(chatId, speech.buffer, {}, { filename: 'reply.mp3', contentType: 'audio/mpeg' });
  } else {
    // Fall back to text if TTS fails, so the user still gets the answer.
    await bot.sendMessage(chatId, reply);
    return;
  }

  // If a web search ran this turn, always send a text companion with the
  // actual names/links — voice alone can't be tapped or read carefully,
  // and Claude often paraphrases search results without a literal URL.
  if (namedResults.length) {
    const lines = namedResults
      .slice(0, 5)
      .map((r) => (r.url ? `• ${r.title} — ${r.url}` : `• ${r.title}`));
    const linkText = '🔗 From that search:\n' + lines.join('\n');
    await bot.sendMessage(chatId, linkText);
  } else if (urls.length) {
    // No search results tracked, but the reply text itself contains links
    // (e.g. from memory or a prior turn) — still worth sending as text.
    const linkText = '🔗 Links from that:\n' + urls.map((u) => `• ${u}`).join('\n');
    await bot.sendMessage(chatId, linkText);
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
      const { reply, searchResults } = await callClaude(
        chatId,
        `[The user just shared their location: ${locationText}. Continue helping with whatever you needed the location for.]`,
        wasVoiceBeforeLocation
      );
      if (reply) {
        await sendReply(chatId, reply, wasVoiceBeforeLocation, searchResults);
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
    registerWithJarvis(msg.from?.username, msg.from?.first_name);
    bot.sendMessage(
      chatId,
      "Hey, I'm Jarvis. Just talk to me normally — I'll remember things you tell me and bring them up later. What's up?"
    );
    sendLocationRequest(chatId, 'know your timezone and location for things like reminders, nearby suggestions, and local weather', false);
    return;
  }

  if (text === '/forget_everything') {
    saveUserData(chatId, emptyUserData());
    bot.sendMessage(chatId, "Done — wiped everything I remembered about you.");
    return;
  }

  if (text === '/memories') {
    const userData = loadUserData(chatId);
    if (!userData.memories.length) {
      bot.sendMessage(chatId, "I don't have anything saved about you yet.");
      return;
    }
    const byCategory = {};
    for (const m of userData.memories) {
      (byCategory[m.category] = byCategory[m.category] || []).push(m);
    }
    const lines = Object.entries(byCategory)
      .map(([cat, items]) => `*${cat}*\n` + items.map((m) => `• ${m.text}`).join('\n'))
      .join('\n\n');
    bot.sendMessage(chatId, `Here's everything I have stored on you (${userData.memories.length} total):\n\n${lines}`, { parse_mode: 'Markdown' });
    return;
  }

  bot.sendChatAction(chatId, wasVoice ? 'record_voice' : 'typing');

  try {
    const { reply, searchResults, anthropicTokens: at1, eventCounts: ec1 } = await callClaude(chatId, text, wasVoice, msg.from?.username || '');
    reportUsage(msg.from?.username, at1 || 0, 0, { ...ec1, messages: 1, voice: wasVoice ? 1 : 0 });
    await sendReply(chatId, reply, wasVoice, searchResults);
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
function nextOccurrence(whenIso, recurrence) {
  const d = new Date(whenIso);
  if (recurrence === 'yearly') d.setFullYear(d.getFullYear() + 1);
  else if (recurrence === 'monthly') d.setMonth(d.getMonth() + 1);
  else if (recurrence === 'weekly') d.setDate(d.getDate() + 7);
  return d.toISOString();
}

async function sendReminderWithSuggestions(chatId, reminder) {
  let messageText = `⏰ Reminder: ${reminder.text}`;

  if (reminder.isGiftOccasion && TAVILY_KEY) {
    try {
      const searchResult = await webSearch(`thoughtful gift ideas for ${reminder.text}`);
      if (searchResult && !searchResult.error && searchResult.results && searchResult.results.length) {
        const ideas = searchResult.results
          .filter((r) => r.url) // only show ideas that have a real, validated link
          .slice(0, 3)
          .map((r) => `• ${r.title} (${r.url})`)
          .join('\n');
        if (ideas) {
          messageText += `\n\nA few gift ideas:\n${ideas}`;
        }
      }
    } catch (err) {
      console.error('Gift suggestion search failed:', err.message);
    }
  }

  return bot.sendMessage(chatId, messageText);
}

async function checkReminders() {
  let files;
  try {
    files = fs.readdirSync(DATA_DIR).filter((f) => f.startsWith('user_') && f.endsWith('.json') && !f.includes('.corrupt-') && !f.endsWith('.tmp'));
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

    // Keep non-due reminders, then re-add recurring ones rescheduled forward.
    const remaining = userData.reminders.filter((r) => new Date(r.when).getTime() > now);
    for (const reminder of due) {
      if (reminder.recurrence && reminder.recurrence !== 'none') {
        remaining.push({ ...reminder, when: nextOccurrence(reminder.when, reminder.recurrence) });
      }
    }
    userData.reminders = remaining;
    saveUserData(chatId, userData);

    for (const reminder of due) {
      sendReminderWithSuggestions(chatId, reminder).catch((err) => {
        console.error('Failed to send reminder to', chatId, err.message);
      });
    }
  }
}

setInterval(checkReminders, 60 * 1000);

// ---------- Webhook server (Vapi call outcomes) ----------
// Vapi posts here when an outbound booking call finishes. We look up which
// Telegram chat requested that call and relay the outcome to them.

function summarizeCallOutcome(message) {
  // Vapi's end-of-call-report includes a "summary" (AI-generated) and/or
  // an "endedReason" describing how the call concluded.
  const summary = message.summary || message.analysis?.summary || null;
  const endedReason = message.endedReason;

  if (summary) return summary;

  if (endedReason === 'customer-did-not-answer' || endedReason === 'no-answer') {
    return "Nobody picked up. Want me to try again later?";
  }
  if (endedReason === 'customer-busy' || endedReason === 'busy') {
    return "The line was busy. Want me to try again?";
  }
  if (endedReason && endedReason.includes('error')) {
    return "The call failed to connect — might be worth double-checking the phone number.";
  }
  return endedReason ? `The call ended (${endedReason}), but I didn't get a clear summary.` : "The call finished, but I didn't get a clear summary.";
}

const webhookServer = http.createServer((req, res) => {
  // Railway health check
  if (req.method === 'GET' && (req.url === '/' || req.url === '/health')) {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('OK');
    return;
  }
  if (req.method === 'POST' && req.url === '/vapi-webhook') {
    let body = '';
    req.on('data', (chunk) => { body += chunk; });
    req.on('end', async () => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true }));

      try {
        const payload = JSON.parse(body);
        const message = payload.message || payload;

        // Vapi sends several event types to the same URL during a call's
        // lifecycle; we only care about the final report.
        if (message.type !== 'end-of-call-report') return;

        const callId = message.call?.id;
        if (!callId) return;

        const calls = loadActiveCalls();
        const callInfo = calls[callId];
        if (!callInfo) {
          console.warn('Received Vapi webhook for unknown call id:', callId);
          return;
        }

        const outcome = summarizeCallOutcome(message);
        const text = `📞 Update on the call to ${callInfo.business_name} (${callInfo.date_time_description}):\n\n${outcome}`;
        await bot.sendMessage(callInfo.chatId, text);

        // Clean up — this call is resolved.
        delete calls[callId];
        saveActiveCalls(calls);
      } catch (err) {
        console.error('Error handling Vapi webhook:', err.message);
      }
    });
    return;
  }

  res.writeHead(404);
  res.end();
});


async function registerWithJarvis(username, firstName) {
  try {
    await fetch('https://c79b1d1c-b690-42a4-89c1-7aa003a55a66-00-3gtw2r50e421s.pike.replit.dev/api/bot/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ telegram_username: username || '', first_name: firstName || '' })
    });
  } catch (_) {}
}

const WEBHOOK_PORT = process.env.PORT || 3000;
webhookServer.listen(WEBHOOK_PORT, () => {
  console.log(`Webhook server listening on port ${WEBHOOK_PORT}`);
});


// ---------- One-time migration: register existing users in admin panel ----------

async function reportUsage(username, anthropicTokens, openaiTokens, events = {}) {
  if (!username) return;
  try {
    await fetch('https://c79b1d1c-b690-42a4-89c1-7aa003a55a66-00-3gtw2r50e421s.pike.replit.dev/api/bot/usage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ telegram_username: username, anthropic_tokens: anthropicTokens, openai_tokens: openaiTokens, events })
    });
  } catch (_) {}
}

async function migrateExistingUsers() {
  let files;
  try {
    files = fs.readdirSync(DATA_DIR).filter(f =>
      f.startsWith('user_') && f.endsWith('.json') &&
      !f.includes('.corrupt-') && !f.endsWith('.tmp')
    );
  } catch (e) { return; }

  console.log('[migrate] Found', files.length, 'user files to sync...');
  let synced = 0;
  let skipped = 0;
  for (const file of files) {
    const chatId = file.slice('user_'.length, -'.json'.length);
    try {
      // Read stored identity directly from file — no Telegram API call needed
      let username = '';
      let firstName = '';
      try {
        const raw = JSON.parse(fs.readFileSync(path.join(DATA_DIR, file), 'utf8'));
        username = raw.username || '';
        firstName = raw.firstName || '';
      } catch (_) {}

      // Fall back to Telegram API only if file has no identity stored
      if (!username && !firstName) {
        try {
          const chat = await bot.getChat(chatId);
          username = chat.username || '';
          firstName = chat.first_name || '';
          await new Promise(r => setTimeout(r, 300)); // rate limit only when needed
        } catch (_) {}
      }

      if (username || firstName) {
        await registerWithJarvis(username, firstName);
        const label = username ? '@' + username : firstName;
        console.log('[migrate] Synced', label, '(chatId=' + chatId + ')');
        synced++;
      } else {
        skipped++;
      }
    } catch (e) {
      console.warn('[migrate] Error for chatId', chatId, e.message);
      skipped++;
    }
  }
  console.log('[migrate] Done. Synced:', synced, 'Skipped:', skipped);
}
migrateExistingUsers();

console.log('Jarvis Telegram bot is running...');
