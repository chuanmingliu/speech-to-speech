import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Text,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type ViewId = "map" | "path" | "machine" | "boot";
type MachineKind = "protocol" | "unit";

type FileRef = {
  label: string;
  path: string;
  startLine?: number;
};

type Block = {
  id: string;
  title: string;
  hint: string;
  lane: string;
  laneNote: string;
  role: string;
  mechanism: string;
  built: string;
  wired: string;
  invariant: string;
  files: FileRef[];
};

type PathStep = {
  title: string;
  hint: string;
  live: string[];
  what: string;
  wire: string;
};

type Scenario = {
  id: string;
  title: string;
  hint: string;
  steps: PathStep[];
};

type MachineState = {
  id: string;
  title: string;
  hint: string;
  enters: string;
  leaves: string;
  code: string;
  files: FileRef[];
};

type BootStep = {
  n: number;
  title: string;
  hint: string;
  ctor: string;
  who: string;
  mustNot: string;
  files: FileRef[];
};

const VIEWS: { id: ViewId; label: string }[] = [
  { id: "map", label: "System map" },
  { id: "path", label: "Request path" },
  { id: "machine", label: "Protocol machine" },
  { id: "boot", label: "Boot wiring" },
];

const BLOCKS: Block[] = [
  {
    id: "you",
    title: "You",
    hint: "OpenAI Realtime client",
    lane: "Client",
    laneNote: "One assistant. Protocol events only.",
    role: "A voice you talk to over a standard Realtime socket. The product looks like one assistant, not a cascade of models.",
    mechanism:
      "Any OpenAI Realtime-compatible client connects to ws://host:8765/v1/realtime (or POSTs SDP to /v1/realtime/calls). Audio is base64 PCM on WebSocket, or Opus on WebRTC. JSON events are the only control surface.",
    built: "No server object. Clients: scripts/listen_and_play_realtime.py, the OpenAI SDK realtime.connect(), or any GA-compatible app.",
    wired: "Client never sees pipeline index, CancelScope.generation, turn_id, turn_revision, or SESSION_END. Those stay inside the unit.",
    invariant: "Do not teach the speech client Session IDs, handler threads, or which STT/LLM/TTS backend is running.",
    files: [
      { label: "Realtime protocol", path: "src/speech_to_speech/api/openai_realtime/README.md" },
      { label: "Listen client", path: "scripts/listen_and_play_realtime.py", startLine: 334 },
    ],
  },
  {
    id: "cli",
    title: "Process entry",
    hint: "speech-to-speech → main()",
    lane: "Process",
    laneNote: "Composition root. Builds everything once.",
    role: "Turns CLI flags or a JSON profile into a running process with one stop_event and N isolated voice graphs.",
    mechanism:
      "setuptools maps speech-to-speech to s2s_pipeline:main. main() parses dataclasses, applies platform presets, then build_pipeline() chooses local / websocket / socket / realtime.",
    built: "main() in s2s_pipeline.py. parse_arguments() returns ParsedArguments. No class — functions compose the graph.",
    wired: "HfArgumentParser registers one LLM argument class after a pre-parse of --llm_backend so responses-api and chat-completions do not collide on shared fields.",
    invariant: "Credentials stay in the environment. JSON profiles hold provider choices, not keys.",
    files: [
      { label: "CLI entry", path: "pyproject.toml", startLine: 111 },
      { label: "main()", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 1027 },
      { label: "parse_arguments()", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 129 },
    ],
  },
  {
    id: "server",
    title: "Realtime server",
    hint: "RealtimeServer + uvicorn",
    lane: "Process",
    laneNote: "One HTTP process. Many pipeline units.",
    role: "Exposes the OpenAI-compatible door: /v1/realtime, optional /v1/realtime/calls, plus /v1/pool and /v1/usage.",
    mechanism:
      "RealtimeServer.run() calls create_app(pool, stop_event) and starts uvicorn. FastAPI lifespan spawns one async _send_loop_for per unit. A stop_event watcher sets server.should_exit.",
    built: "RealtimeServer(stop_event, pool, host, port) constructed in build_pipeline() when mode is realtime. create_app() is a factory, not a class.",
    wired: "The server thread is just another ThreadManager handler. It must not own handler threads; those are siblings.",
    invariant: "One uvicorn for the process. Extra connections are rejected with session_limit_reached — they do not spawn extra models.",
    files: [
      { label: "RealtimeServer", path: "src/speech_to_speech/api/openai_realtime/server.py", startLine: 13 },
      { label: "create_app()", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 406 },
      { label: "build_pipeline realtime", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 655 },
    ],
  },
  {
    id: "unit",
    title: "Pipeline unit",
    hint: "PipelineUnit + SessionState",
    lane: "Per connection",
    laneNote: "Claimed on accept. Released after SESSION_END drains.",
    role: "Gives each client its own queues, CancelScope, RealtimeService, and handler chain so two conversations cannot share audio or chat.",
    mechanism:
      "_claim_unit() reserves the first unit with session is None. register() mints a session_id. On disconnect, SESSION_END walks the handler chain; the send loop sets drained; only then is session cleared.",
    built: "PipelineUnit(...) returned by _build_realtime_pipeline_unit(index, ...). SessionState(transport=...) created inside _claim_unit.",
    wired: "unit.session is the claim lock. The send loop snapshots session each tick so a mid-iteration release cannot leak audio to the next client.",
    invariant: "A new client must not claim a unit until SESSION_END drains. Quarantine beats cross-session leak.",
    files: [
      { label: "PipelineUnit", path: "src/speech_to_speech/api/openai_realtime/pipeline_unit.py", startLine: 49 },
      { label: "SessionState", path: "src/speech_to_speech/api/openai_realtime/pipeline_unit.py", startLine: 13 },
      { label: "_claim_unit", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 430 },
    ],
  },
  {
    id: "service",
    title: "Protocol adapter",
    hint: "RealtimeService",
    lane: "Per connection",
    laneNote: "One protocol client for this unit's backend chain.",
    role: "Translates OpenAI Realtime events into pipeline messages and back. The cascade never speaks protocol names.",
    mechanism:
      "parse_client_event() validates against openai.types.realtime models. AudioHandler / SessionHandler / ConversationHandler / ResponseHandler own inbound events. dispatch_pipeline_event() maps SpeechStartedEvent, transcription, AssistantTextEvent to server events.",
    built: "RealtimeService(text_prompt_queue, should_listen, chat_size, speculative_turns) inside _build_realtime_pipeline_unit. Handlers are constructed as AudioHandler(self), SessionHandler(self), …",
    wired: "ConnState is keyed by session_id from register(). RuntimeConfig.chat is the conversation the LLM actually reads.",
    invariant: "conversation.item.create does not generate. response.create does. Do not invent DashScope event names here.",
    files: [
      { label: "RealtimeService", path: "src/speech_to_speech/api/openai_realtime/service.py", startLine: 191 },
      { label: "Client event map", path: "src/speech_to_speech/api/openai_realtime/service.py", startLine: 69 },
      { label: "_dispatch_client_event", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 317 },
    ],
  },
  {
    id: "vad",
    title: "VAD",
    hint: "speech_started / speech_stopped",
    lane: "Voice graph",
    laneNote: "Must stay live while LLM and TTS work.",
    role: "Decides when you started and stopped talking, and whether that barge-in should cancel the current spoken reply.",
    mechanism:
      "Consumes 16 kHz / 512-sample PCM from recv_audio_chunks_queue. Silero VAD v5 emits SpeechStartedEvent / SpeechStoppedEvent on text_output_queue and a VADAudio utterance on spoken_prompt_queue.",
    built: "VADHandler(stop_event, queue_in=recv, queue_out=spoken, setup_args=(should_listen,), setup_kwargs=vars(vad_kw)). Instantiated first in _build_pipeline_handlers.",
    wired: "setup() receives text_output_queue and speculative_turns. interrupt_response is stamped on SpeechStartedEvent; the send loop AND RuntimeConfig.interrupt_response_enabled must both allow cancel.",
    invariant: "Listening is not paused for generation. should_listen gates capture; the VAD thread itself never joins the LLM.",
    files: [
      { label: "VADHandler", path: "src/speech_to_speech/VAD/vad_handler.py", startLine: 53 },
      { label: "VADIterator", path: "src/speech_to_speech/VAD/vad_iterator.py" },
    ],
  },
  {
    id: "stt",
    title: "STT",
    hint: "Transcription(turn_id, turn_revision)",
    lane: "Voice graph",
    laneNote: "Adapter boundary. Provider encoding stays here.",
    role: "Turns a finalized (or speculative) utterance into text the rest of the graph can share.",
    mechanism:
      "get_stt_handler() picks Parakeet, Whisper, Faster Whisper, Paraformer, Tencent, or MLX Whisper. Output is Transcription or PartialTranscription. Live partials can stream while VAD is still open.",
    built: "Factory get_stt_handler(module_kwargs, …) news the concrete BaseSTTHandler. with_speculative_turns() assigns the unit tracker when present.",
    wired: "queue_in = spoken_prompt_queue, queue_out = stt_output_queue. Tencent realtime ASR flips enable_live_transcription when TENCENT_ASR_APP_ID is set.",
    invariant: "Preserve turn_id and turn_revision across the adapter. Do not retry a stale revision after barge-in.",
    files: [
      { label: "get_stt_handler()", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 767 },
      { label: "BaseSTTHandler", path: "src/speech_to_speech/STT/base_stt_handler.py" },
      { label: "Parakeet (default)", path: "src/speech_to_speech/STT/parakeet_tdt_handler.py" },
    ],
  },
  {
    id: "notifier",
    title: "Write it down",
    hint: "TranscriptionNotifier",
    lane: "Voice graph",
    laneNote: "Taps transcripts. Does not own generation in realtime.",
    role: "Publishes transcription.delta / transcription.completed to the client without letting STT speak protocol.",
    mechanism:
      "In realtime mode (no runtime_config) it puts PartialTranscriptionEvent / TranscriptionCompletedEvent on text_output_queue and yields nothing. RealtimeService._on_transcription_completed appends Chat and may enqueue GenerateResponseRequest.",
    built: "TranscriptionNotifier(stop_event, queue_in=stt_output, queue_out=text_prompt, setup_kwargs={text_output_queue, should_listen}).",
    wired: "Legacy local/socket modes pass runtime_config so the notifier itself appends Chat and yields GenerateResponseRequest.",
    invariant: "Empty finals still emit completed so clients can close a partial item. Empty finals must not start the LLM.",
    files: [
      { label: "TranscriptionNotifier", path: "src/speech_to_speech/STT/transcription_notifier.py", startLine: 19 },
      { label: "STT → LM bridge", path: "src/speech_to_speech/api/openai_realtime/service.py", startLine: 407 },
    ],
  },
  {
    id: "llm",
    title: "LLM",
    hint: "GenerateResponseRequest",
    lane: "Voice graph",
    laneNote: "OpenAI-compatible slot. Hosted or local.",
    role: "Produces the assistant's next words and optional tool calls from Chat, not from raw audio.",
    mechanism:
      "get_llm_handler() news ResponsesApiModelHandler, ChatCompletionsApiModelHandler, LanguageModelHandler, or VisionLanguageModelHandler. Each yields LLMResponseChunk / TokenUsage / EndOfResponse. Local tools are parsed from <code> blocks; API tools arrive as native function_call items.",
    built: "Factory in get_llm_handler(). Constructor: Handler(stop_event, queue_in=text_prompt_queue, queue_out=lm_response_queue, setup_kwargs=vars(lm_kw)).",
    wired: "cancel_scope and speculative_turns are injected into kwargs before construction. Handlers capture generation at response start and abort on is_stale(gen).",
    invariant: "The model backend must not learn pipeline unit index, SESSION_END, or how tools are executed. It only sees Chat + tools schema.",
    files: [
      { label: "get_llm_handler()", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 878 },
      { label: "Responses API", path: "src/speech_to_speech/LLM/responses_api_language_model.py" },
      { label: "Local LanguageModelHandler", path: "src/speech_to_speech/LLM/language_model.py" },
    ],
  },
  {
    id: "tools",
    title: "Tool split",
    hint: "LMOutputProcessor",
    lane: "Voice graph",
    laneNote: "Spoken text one way. Tool dicts the other.",
    role: "Keeps tool calls off the speaker. The client executes tools; the pipeline only announces them.",
    mechanism:
      "Reads LLMResponseChunk. Puts AssistantTextEvent (optional .tools) on text_output_queue. Yields TTSInput only when response_wants_audio and text is non-empty.",
    built: "LMOutputProcessor(stop_event, queue_in=lm_response, queue_out=lm_processed, setup_kwargs={text_output_queue, speculative_turns}).",
    wired: "ResponseHandler.on_assistant_text turns tools into response.function_call_arguments.done. Client returns via conversation.item.create {type: function_call_output}. That does not speak until response.create.",
    invariant: "Tool output is context, not a trigger. Fire-and-forget robot actions can stop after conversation.item.created.",
    files: [
      { label: "LMOutputProcessor", path: "src/speech_to_speech/LLM/lm_output_processor.py", startLine: 26 },
      { label: "ConversationHandler", path: "src/speech_to_speech/api/openai_realtime/handlers/conversation.py", startLine: 27 },
      { label: "ResponseHandler tools", path: "src/speech_to_speech/api/openai_realtime/handlers/response.py", startLine: 33 },
    ],
  },
  {
    id: "tts",
    title: "TTS",
    hint: "AudioOutput + AUDIO_RESPONSE_DONE",
    lane: "Voice graph",
    laneNote: "Sentence-sized input. Chunked PCM out.",
    role: "Turns clean text into playable audio the send loop can encode as response.output_audio.delta.",
    mechanism:
      "get_tts_handler() news Qwen3, Kokoro, Pocket, ChatTTS, MMS, or MiniMax. Each yields int16 PCM chunks tagged with cancel_generation, then a done sentinel.",
    built: "Factory get_tts_handler(..., cancel_scope, speculative_turns). Most handlers take setup_args=(should_listen,). MiniMax receives cancel_scope in setup_kwargs.",
    wired: "queue_in = lm_processed_queue, queue_out = send_audio_chunks_queue. should_listen is set again when the send loop sees the done sentinel.",
    invariant: "Check CancelScope and SpeculativeTurnTracker before every yield. Do not play a superseded revision.",
    files: [
      { label: "get_tts_handler()", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 935 },
      { label: "Qwen3 (default)", path: "src/speech_to_speech/TTS/qwen3_tts_handler.py" },
      { label: "MiniMax adapter", path: "src/speech_to_speech/TTS/minimax_tts_handler.py" },
    ],
  },
  {
    id: "config",
    title: "Session config",
    hint: "RuntimeConfig + Chat",
    lane: "Control",
    laneNote: "Written on session.update. Read at process time.",
    role: "Holds the live instructions, tools, voice, and turn-detection the handlers read without restarting threads.",
    mechanism:
      "session.update deep-merges into RuntimeConfig.session. VAD reads interrupt_response. LLM reads instructions and tools. TTS reads voice. Chat is the rolling conversation window.",
    built: "RuntimeConfig(chat=Chat(chat_size)) inside ConnState, created by RealtimeService.register(). Chat(size) is also constructed for legacy notifier setup.",
    wired: "Python GIL makes primitive field reads atomic. No lock. Handlers receive the current RuntimeConfig on the message, not by importing the socket.",
    invariant: "Partial updates must not wipe unset nested fields. Transcription-only session types are rejected (invalid_session_type).",
    files: [
      { label: "RuntimeConfig", path: "src/speech_to_speech/api/openai_realtime/runtime_config.py", startLine: 27 },
      { label: "Chat", path: "src/speech_to_speech/LLM/chat.py", startLine: 50 },
      { label: "SessionHandler", path: "src/speech_to_speech/api/openai_realtime/handlers/session.py", startLine: 20 },
    ],
  },
  {
    id: "cancel",
    title: "Cancel scope",
    hint: "generation++ / discarding",
    lane: "Control",
    laneNote: "Interruption cancels speech, not the listen path.",
    role: "Lets barge-in stop the current spoken reply without tearing down the unit or dropping user transcripts still in flight.",
    mechanism:
      "cancel() increments generation and sets discarding. LLM/TTS abort when is_stale(captured_gen). The send loop drops tagged output from old generations. response_done(gen) or new_response() clears the discard window.",
    built: "CancelScope() in _build_realtime_pipeline_unit. One per unit. Also created in initialize_queues_and_events() for non-realtime modes.",
    wired: "Writer is the asyncio send loop / _dispatch_client_event. Readers are handler threads. No lock — int/bool writes are atomic under the GIL.",
    invariant: "Flush preserves SESSION_END and user-side text events (speech_stopped, transcriptions, usage). Never cancel() when no response is active.",
    files: [
      { label: "CancelScope", path: "src/speech_to_speech/pipeline/cancel_scope.py", startLine: 1 },
      { label: "Barge-in in send loop", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 742 },
      { label: "response.cancel", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 393 },
    ],
  },
  {
    id: "spec",
    title: "Speculative turns",
    hint: "turn_id / turn_revision",
    lane: "Control",
    laneNote: "Newer revisions win. Old speech dies.",
    role: "Lets STT emit a first-pass transcript and later replace it without the stale version reaching playback.",
    mechanism:
      "VAD/STT stamp turn_id and increasing turn_revision. SpeculativeTurnTracker.observe() records the latest. LMOutputProcessor and RealtimeService drop events that are not latest after reopen grace.",
    built: "SpeculativeTurnTracker() in _build_realtime_pipeline_unit. Assigned onto VAD kwargs, STT handler, LLM/TTS kwargs, RealtimeService, and LMOutputProcessor.",
    wired: "register() calls speculative_turns.reset() so a new session cannot inherit the previous client's revisions.",
    invariant: "Do not replay ASR or TTS after a newer revision exists. Out-of-band response.create uses turn_id=None so it is never treated as stale.",
    files: [
      { label: "SpeculativeTurnTracker", path: "src/speech_to_speech/pipeline/speculative_turns.py", startLine: 24 },
      { label: "Stale event gate", path: "src/speech_to_speech/api/openai_realtime/service.py", startLine: 376 },
    ],
  },
  {
    id: "send",
    title: "Send loop",
    hint: "_send_loop_for",
    lane: "Wire",
    laneNote: "Sole consumer of output queues.",
    role: "The only place pipeline bytes become protocol audio. Keeps the client clocked even when handlers burst.",
    mechanism:
      "Async loop per unit. Text events first (so speech_started can cancel). Then audio: batch PCM, encode response.output_audio.delta, finish on AUDIO_RESPONSE_DONE. SESSION_END sets session.drained.",
    built: "Not a class. asyncio.create_task(_send_loop_for(unit)) in FastAPI lifespan inside create_app().",
    wired: "Reads unit.session.transport. WebSocket encodes base64 PCM. WebRTC paces 20 ms RTP and supports output_audio_buffer.clear.",
    invariant: "One consumer per unit. Handler threads must not send on the socket. Current-generation audio must never be swallowed by a lingering discard window.",
    files: [
      { label: "_send_loop_for", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 703 },
      { label: "AudioHandler.encode", path: "src/speech_to_speech/api/openai_realtime/handlers/audio.py", startLine: 31 },
    ],
  },
  {
    id: "threads",
    title: "Thread manager",
    hint: "ThreadManager.start()",
    lane: "Process",
    laneNote: "One OS thread per handler, plus the server.",
    role: "Keeps VAD, STT, LLM, TTS, and uvicorn alive together so a slow model cannot block the microphone path.",
    mechanism:
      "start() spawns threading.Thread(target=handler.run) for RealtimeServer and every unit handler. stop() sets stop_event and joins with a 5s timeout.",
    built: "ThreadManager([realtime_server, *unit.handlers]) in build_pipeline() realtime branch. Non-realtime: ThreadManager([*comms, *pipeline_handlers]).",
    wired: "BaseHandler.run() loops queue_in → process() → queue_out until stop_event. SESSION_END is a soft reset; PIPELINE_END is process death.",
    invariant: "Handlers must not import uvicorn or claim units. The server must not process audio.",
    files: [
      { label: "ThreadManager", path: "src/speech_to_speech/utils/thread_manager.py", startLine: 9 },
      { label: "BaseHandler", path: "src/speech_to_speech/baseHandler.py", startLine: 23 },
    ],
  },
];

const BLOCK_BY_ID: Record<string, Block> = Object.fromEntries(BLOCKS.map((b) => [b.id, b]));

const LANES: { id: string; note: string; blocks: string[] }[] = [
  { id: "Client", note: "What the user thinks they are talking to.", blocks: ["you"] },
  { id: "Process", note: "Built once at boot. Shared by every connection.", blocks: ["cli", "server", "threads"] },
  { id: "Per connection", note: "Claimed on accept. Isolated queues and Chat.", blocks: ["unit", "service"] },
  {
    id: "Voice graph",
    note: "VAD → STT → notifier → LLM → tool split → TTS",
    blocks: ["vad", "stt", "notifier", "llm", "tools", "tts"],
  },
  { id: "Control", note: "Stay live while work runs.", blocks: ["config", "cancel", "spec"] },
  { id: "Wire", note: "Protocol in. Protocol out.", blocks: ["send"] },
];

const SCENARIOS: Scenario[] = [
  {
    id: "spoken",
    title: "Spoken turn",
    hint: "input_audio_buffer.append → output_audio.delta",
    steps: [
      {
        title: "Connect",
        hint: "session.created",
        live: ["you", "server", "unit", "service", "send"],
        what: "Client opens /v1/realtime. The router claims a free PipelineUnit, register() mints ConnState, and session.created goes out.",
        wire: "create_app → _claim_unit → RealtimeService.register → build_session_created",
      },
      {
        title: "Setup",
        hint: "session.update",
        live: ["you", "service", "config", "vad", "llm", "tts"],
        what: "Instructions, tools, voice, and server_vad.interrupt_response deep-merge into RuntimeConfig. Handler threads keep running.",
        wire: "SessionHandler.handle_session_update → RuntimeConfig.apply_session_update",
      },
      {
        title: "You talk",
        hint: "speech_started",
        live: ["you", "service", "vad", "send"],
        what: "append events decode to 16 kHz / 512-sample PCM on input_queue. VAD emits SpeechStartedEvent. Send loop translates it first, before audio.",
        wire: "AudioHandler.handle_audio_append → input_queue → VADHandler → text_output_queue",
      },
      {
        title: "Write it down",
        hint: "transcription.completed",
        live: ["vad", "stt", "notifier", "service", "config"],
        what: "Utterance audio hits STT. Notifier publishes completed. Service appends a user Chat item and, for a non-empty transcript, enqueues GenerateResponseRequest.",
        wire: "TranscriptionNotifier → RealtimeService._on_transcription_completed → text_prompt_queue",
      },
      {
        title: "Think",
        hint: "response.created",
        live: ["llm", "config", "cancel"],
        what: "The LLM handler reads Chat, streams tokens, and stamps cancel_generation. First outbound audio (or explicit response.create) marks the response in_progress.",
        wire: "LLM handler → lm_response_queue → LMOutputProcessor",
      },
      {
        title: "Speak",
        hint: "response.output_audio.delta",
        live: ["tools", "tts", "send", "you"],
        what: "Clean text becomes TTSInput. PCM lands on output_queue. Send loop batches and encodes deltas, then finish_response on AUDIO_RESPONSE_DONE.",
        wire: "TTS → send_audio_chunks_queue → _send_loop_for → transport.send_events",
      },
    ],
  },
  {
    id: "tool",
    title: "Tool round-trip",
    hint: "function_call_arguments.done",
    steps: [
      {
        title: "Model asks",
        hint: "function_call_arguments.done",
        live: ["llm", "tools", "send", "you"],
        what: "LLM yields tools on LLMResponseChunk. LMOutputProcessor copies them onto AssistantTextEvent. ResponseHandler emits one function_call_arguments.done per call.",
        wire: "LMOutputProcessor.process → text_output_queue → ResponseHandler.on_assistant_text",
      },
      {
        title: "Client runs it",
        hint: "conversation.item.create",
        live: ["you", "service", "config"],
        what: "Client sends function_call_output. ConversationHandler appends Chat (or defers if a response is still generating). conversation.item.created acknowledges. No generation.",
        wire: "ConversationHandler.handle_conversation_item_create → Chat.add_item",
      },
      {
        title: "Ask it to speak",
        hint: "response.create",
        live: ["you", "service", "llm", "cancel", "config"],
        what: "Only response.create puts GenerateResponseRequest on text_prompt_queue and calls cancel_scope.new_response() so a leftover discard window cannot swallow the follow-up.",
        wire: "ResponseHandler.handle_response_create → text_prompt_queue.put(GenerateResponseRequest)",
      },
      {
        title: "Spoken result",
        hint: "output_audio.delta",
        live: ["llm", "tools", "tts", "send", "you"],
        what: "Follow-up text is split again: tools stay on the side channel, spoken sentences go to TTS.",
        wire: "Same tool-split path as a normal turn",
      },
    ],
  },
  {
    id: "barge",
    title: "You interrupt",
    hint: "speech_started + CancelScope.cancel",
    steps: [
      {
        title: "Reply is playing",
        hint: "in_response / response_pending",
        live: ["tts", "send", "you", "cancel"],
        what: "TTS is writing PCM. Send loop is encoding deltas. ConnState.in_response is true, or response_pending if create was accepted but audio has not started.",
        wire: "response_playing Event + ConnState.in_response",
      },
      {
        title: "VAD hears you",
        hint: "speech_started",
        live: ["vad", "send", "you"],
        what: "SpeechStartedEvent is processed before queued audio. Client gets speech_started, then response.done status=cancelled reason=turn_detected.",
        wire: "dispatch_pipeline_event → AudioHandler.on_speech_started → finish_response",
      },
      {
        title: "Generation dies",
        hint: "generation++",
        live: ["cancel", "llm", "tts", "send", "spec"],
        what: "If interrupt_response is enabled, cancel() increments generation and discarding=True. Queues flush but keep SESSION_END and user transcripts. LLM/TTS abort stale gens.",
        wire: "CancelScope.cancel → _flush_queue(preserve=sentinels/user events)",
      },
      {
        title: "Listen continues",
        hint: "should_listen",
        live: ["vad", "stt", "notifier", "unit"],
        what: "The same unit keeps listening. The new utterance is a new turn_revision. Old audio cannot play once its generation is stale.",
        wire: "should_listen.set on stale/current AUDIO_RESPONSE_DONE",
      },
    ],
  },
  {
    id: "inject",
    title: "Injected text",
    hint: "conversation.item.create then response.create",
    steps: [
      {
        title: "Add context",
        hint: "conversation.item.create",
        live: ["you", "service", "config"],
        what: "input_text or function_call_output is appended to Chat. If a response is generating, the item is deferred so it cannot race the LLM thread's write-back.",
        wire: "ConversationHandler — defer while in_response, else _apply_item",
      },
      {
        title: "Generate",
        hint: "response.create",
        live: ["service", "llm", "cancel"],
        what: "response.create is the only generate signal. A second in-band create while in_response returns conversation_already_has_active_response.",
        wire: "ResponseHandler.handle_response_create",
      },
      {
        title: "Speak or not",
        hint: "response_wants_audio",
        live: ["tools", "tts", "send"],
        what: "LMOutputProcessor forwards text to TTS only when the response modalities include audio. Text-only responses skip the speaker.",
        wire: "response_wants_audio(lm_output.response)",
      },
    ],
  },
];

const PROTOCOL_STATES: MachineState[] = [
  {
    id: "off",
    title: "Off",
    hint: "no socket",
    enters: "Process is up, pool units are idle (session is None).",
    leaves: "Client connects to /v1/realtime or POSTs SDP to /v1/realtime/calls.",
    code: "unit = _claim_unit(transport)\nif unit is None: error session_limit_reached",
    files: [
      { label: "Claim", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 430 },
    ],
  },
  {
    id: "opening",
    title: "Opening",
    hint: "session.created",
    enters: "Claim succeeded. register() created ConnState. Queues cleaned.",
    leaves: "session.created is sent (WebSocket immediately; WebRTC when oai-events opens).",
    code: "session_id = unit.service.register()\nawait send_ws_event(ws, unit.service.build_session_created(session_id))",
    files: [
      { label: "register()", path: "src/speech_to_speech/api/openai_realtime/service.py", startLine: 229 },
      { label: "Endpoint", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 443 },
    ],
  },
  {
    id: "setup",
    title: "Setup",
    hint: "session.update",
    enters: "Client deep-merges instructions, tools, voice, turn_detection.",
    leaves: "RuntimeConfig is readable by VAD, LLM, and TTS on the next item.",
    code: "cfg.apply_session_update(s)  # only model_fields_set",
    files: [
      { label: "SessionHandler", path: "src/speech_to_speech/api/openai_realtime/handlers/session.py", startLine: 23 },
    ],
  },
  {
    id: "ready",
    title: "Ready",
    hint: "input_audio_buffer.append",
    enters: "PCM is accepted. WebRTC rejects append (invalid_event_for_transport) and uses the media track instead.",
    leaves: "VAD sees 512-sample chunks on input_queue.",
    code: "chunks = service.handle_audio_append(session_id, event)\nfor chunk in chunks:\n    unit.input_queue.put((chunk, rt_cfg))",
    files: [
      { label: "append", path: "src/speech_to_speech/api/openai_realtime/handlers/audio.py", startLine: 48 },
    ],
  },
  {
    id: "talk",
    title: "You talk",
    hint: "speech_started",
    enters: "Silero crosses min_speech_ms. SpeechStartedEvent hits text_output_queue.",
    leaves: "Client receives input_audio_buffer.speech_started. If a reply was active, cancel path may run.",
    code: "Send loop processes text_msg first.\nSpeechStartedEvent → dispatch_pipeline_event",
    files: [
      { label: "Send loop text-first", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 721 },
    ],
  },
  {
    id: "write",
    title: "Write it down",
    hint: "transcription.completed",
    enters: "STT emits Transcription. Notifier publishes the protocol event.",
    leaves: "Chat gains a user message. Non-empty transcript can enqueue GenerateResponseRequest.",
    code: "cfg.chat.add_item(make_user_message(transcript))\n# empty transcript: completed still fires, LLM does not",
    files: [
      { label: "Notifier", path: "src/speech_to_speech/STT/transcription_notifier.py", startLine: 55 },
      { label: "Bridge", path: "src/speech_to_speech/api/openai_realtime/service.py", startLine: 407 },
    ],
  },
  {
    id: "think",
    title: "Think",
    hint: "response.created",
    enters: "GenerateResponseRequest or first outbound audio. in_response=True.",
    leaves: "Tokens and optional tools appear as LLMResponseChunk.",
    code: "queue.put(GenerateResponseRequest(...))\nreturn ResponseCreatedEvent(response=in_progress)",
    files: [
      { label: "response.create", path: "src/speech_to_speech/api/openai_realtime/handlers/response.py", startLine: 137 },
    ],
  },
  {
    id: "speak",
    title: "Speak",
    hint: "response.output_audio.delta",
    enters: "TTS PCM passes the discard gate and is encoded.",
    leaves: "AUDIO_RESPONSE_DONE → finish_response(completed) → should_listen.set.",
    code: "events = service.encode_audio_chunk(conn_id, audio)\n# then finish_response on done sentinel",
    files: [
      { label: "Encode", path: "src/speech_to_speech/api/openai_realtime/handlers/audio.py", startLine: 31 },
      { label: "Done sentinel", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 787 },
    ],
  },
  {
    id: "tool",
    title: "Call a tool",
    hint: "function_call_arguments.done",
    enters: "AssistantTextEvent.tools is non-empty.",
    leaves: "Client must conversation.item.create the output, then response.create to speak.",
    code: "conversation.item.create  # does not generate\nresponse.create           # does",
    files: [
      { label: "Item create", path: "src/speech_to_speech/api/openai_realtime/handlers/conversation.py", startLine: 30 },
    ],
  },
  {
    id: "interrupt",
    title: "You interrupt",
    hint: "response.cancel / turn_detected",
    enters: "Client response.cancel, or speech_started while in_response and interrupt_response is true.",
    leaves: "generation++, queues flushed with sentinels kept, listening re-enabled. The unit stays claimed.",
    code: "unit.cancel_scope.cancel()\n_flush_queue(output, preserve=SESSION_END)\n_flush_queue(text, preserve=user events)",
    files: [
      { label: "Client cancel", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 393 },
      { label: "CancelScope.cancel", path: "src/speech_to_speech/pipeline/cancel_scope.py", startLine: 24 },
    ],
  },
];

const UNIT_STATES: MachineState[] = [
  {
    id: "idle",
    title: "Idle",
    hint: "session is None",
    enters: "Boot, or previous SESSION_END drained and session was cleared.",
    leaves: "_claim_unit finds this unit first.",
    code: "for unit in pool:\n    if unit.session is None:\n        unit.session = SessionState(transport=transport)",
    files: [
      { label: "Claim", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 430 },
    ],
  },
  {
    id: "active",
    title: "Active",
    hint: "released_at is None",
    enters: "register() assigned session_id. Client is connected.",
    leaves: "WebSocketDisconnect, WebRTC close, or DELETE /v1/realtime/calls/{id}.",
    code: "session_id = unit.service.register()\nunit.session.session_id = session_id",
    files: [
      { label: "Endpoint try", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 468 },
    ],
  },
  {
    id: "draining",
    title: "Draining",
    hint: "SESSION_END in flight",
    enters: "_release_session sets released_at and injects SESSION_END at input_queue.",
    leaves: "Send loop sees SESSION_END for this session_id and sets drained.",
    code: "if is_control_message(audio_chunk, SESSION_END.kind):\n    session.drained.set()",
    files: [
      { label: "Drain observe", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 807 },
      { label: "SESSION_END", path: "src/speech_to_speech/pipeline/control.py" },
    ],
  },
  {
    id: "stuck",
    title: "Stuck",
    hint: "quarantined_at set",
    enters: "SESSION_END_QUARANTINE_TIMEOUT_S (180s) elapsed. Service is unregistered so late output cannot bill or mutate Chat.",
    leaves: "Only if SESSION_END eventually drains. A dead handler keeps the slot occupied forever, visible on /v1/pool.",
    code: "# unit stays unclaimable\n# /v1/pool reports state: stuck",
    files: [
      { label: "Pool states", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 523 },
    ],
  },
  {
    id: "live-gen",
    title: "Live generation",
    hint: "CancelScope._discarding = False",
    enters: "New unit, reset(), response_done(), or new_response() after response.create.",
    leaves: "cancel() on barge-in or response.cancel while a response is active.",
    code: "def new_response(self) -> None:\n    self._discarding = False",
    files: [
      { label: "new_response", path: "src/speech_to_speech/pipeline/cancel_scope.py", startLine: 46 },
    ],
  },
  {
    id: "discarding",
    title: "Discarding",
    hint: "generation++ ",
    enters: "cancel() from the send loop or response.cancel path.",
    leaves: "response_done(matching gen) or new_response(). Unrelated older sentinels are ignored.",
    code: "self._discarded_generation = self._gen\nself._gen = (self._gen + 1) & 0xFFFFFFFF\nself._discarding = True",
    files: [
      { label: "cancel()", path: "src/speech_to_speech/pipeline/cancel_scope.py", startLine: 24 },
    ],
  },
];

const BOOT: BootStep[] = [
  {
    n: 1,
    title: "Console script",
    hint: "speech-to-speech",
    ctor: "setuptools entry point → speech_to_speech.s2s_pipeline:main",
    who: "pip / uv install. No object yet.",
    mustNot: "Must not import handler implementations at module import beyond NLTK warmup and TORCHINDUCTOR_CACHE_DIR.",
    files: [{ label: "pyproject.toml", path: "pyproject.toml", startLine: 111 }],
  },
  {
    n: 2,
    title: "Parse env and flags",
    hint: "parse_arguments()",
    ctor: "HfArgumentParser((ModuleArguments, transports, VAD, STTs, one LM class, TTSs))",
    who: "main() → parse_arguments(). JSON file path is an alternate argv form.",
    mustNot: "Must not register both Responses and ChatCompletions argument classes — shared field names collide.",
    files: [{ label: "parse_arguments", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 129 }],
  },
  {
    n: 3,
    title: "Prepare kwargs",
    hint: "prepare_all_args()",
    ctor: "Mutates dataclasses in place: Mac presets, Tencent live-ASR, device overwrite, rename_args prefixes.",
    who: "main(), after parse, before any handler is constructed.",
    mustNot: "Must not check num_pipelines against mode before this — local_mac_optimal_settings flips mode to local.",
    files: [{ label: "prepare_all_args", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 301 }],
  },
  {
    n: 4,
    title: "Process queues",
    hint: "initialize_queues_and_events()",
    ctor: "Event stop/listen/playing, CancelScope(), and the typed Queue set used by non-realtime modes.",
    who: "main(). Realtime mode still creates this dict, then _build_realtime_pipeline_unit makes per-unit copies.",
    mustNot: "Realtime units must not share these queues — deep-copied kwargs and new Queue() instances only.",
    files: [{ label: "initialize_queues", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 346 }],
  },
  {
    n: 5,
    title: "Per-unit control",
    hint: "CancelScope + SpeculativeTurnTracker",
    ctor: "cancel_scope = CancelScope(); speculative_turns = SpeculativeTurnTracker()",
    who: "_build_realtime_pipeline_unit, once per pool index.",
    mustNot: "Must not reuse a tracker or scope across units. register() resets the tracker on claim.",
    files: [{ label: "unit builder", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 505 }],
  },
  {
    n: 6,
    title: "Protocol adapter",
    hint: "RealtimeService(...)",
    ctor: "RealtimeService(text_prompt_queue, should_listen, chat_size, speculative_turns) then Audio/Session/Response/Conversation handlers(self).",
    who: "_build_realtime_pipeline_unit, before the handler chain.",
    mustNot: "Must not start uvicorn. Must not import STT/TTS backends.",
    files: [{ label: "RealtimeService()", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 537 }],
  },
  {
    n: 7,
    title: "Voice graph",
    hint: "_build_pipeline_handlers()",
    ctor: "new VADHandler, get_stt_handler(), TranscriptionNotifier, get_llm_handler(), LMOutputProcessor, get_tts_handler() — in that order.",
    who: "_build_realtime_pipeline_unit after kwargs received cancel_scope / text_output_queue.",
    mustNot: "Adapters must not import FastAPI or protocol event classes. They speak pipeline messages only.",
    files: [{ label: "handler chain", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 363 }],
  },
  {
    n: 8,
    title: "Wrap the unit",
    hint: "PipelineUnit(...)",
    ctor: "PipelineUnit(index, service, cancel_scope, events, queues, handlers). session defaults to None.",
    who: "_build_realtime_pipeline_unit return. build_pipeline maps range(num_pipelines).",
    mustNot: "Must not create uvicorn here. The comment in the builder says the single server is owned by RealtimeServer.",
    files: [{ label: "PipelineUnit return", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 583 }],
  },
  {
    n: 9,
    title: "HTTP + threads",
    hint: "RealtimeServer + ThreadManager",
    ctor: "RealtimeServer(stop_event, pool, host, port) then ThreadManager([server, *all unit.handlers]).start()",
    who: "build_pipeline realtime branch, then main() start/wait.",
    mustNot: "UIs and clients must not spawn this process. SIGINT only sets stop_event.",
    files: [
      { label: "RealtimeServer()", path: "src/speech_to_speech/s2s_pipeline.py", startLine: 681 },
      { label: "ThreadManager", path: "src/speech_to_speech/utils/thread_manager.py", startLine: 14 },
    ],
  },
  {
    n: 10,
    title: "Accept loop",
    hint: "create_app lifespan",
    ctor: "uvicorn.Server(Config(app)).run() after create_app starts one _send_loop_for task per unit.",
    who: "RealtimeServer.run on its ThreadManager thread.",
    mustNot: "Send loops must not share queues. Claim must fail closed when the pool is full.",
    files: [
      { label: "run()", path: "src/speech_to_speech/api/openai_realtime/server.py", startLine: 36 },
      { label: "lifespan", path: "src/speech_to_speech/api/openai_realtime/websocket_router.py", startLine: 407 },
    ],
  },
];

function openRefs(
  dispatch: (action: { type: "openFile"; path: string; selection?: { startLineNumber: number; startColumn: number } }) => void,
  files: FileRef[],
) {
  return files.map((file) => (
    <span key={file.path + String(file.startLine ?? 0)}>
      <Button
        variant="secondary"
        onClick={() =>
          dispatch({
            type: "openFile",
            path: file.path,
            selection:
              file.startLine !== undefined
                ? { startLineNumber: file.startLine, startColumn: 1 }
                : undefined,
          })
        }
      >
        {file.label}
      </Button>
    </span>
  ));
}

function DiagramNode({
  title,
  hint,
  active,
  dim,
  onClick,
}: {
  title: string;
  hint: string;
  active?: boolean;
  dim?: boolean;
  onClick?: () => void;
}) {
  const theme = useHostTheme();
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        textAlign: "left",
        minWidth: 128,
        maxWidth: 180,
        padding: "8px 10px",
        borderRadius: 6,
        border: `1px solid ${active ? theme.accent.primary : theme.stroke.secondary}`,
        background: active ? theme.fill.tertiary : theme.bg.elevated,
        opacity: dim ? 0.45 : 1,
        cursor: onClick ? "pointer" : "default",
        color: theme.text.primary,
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 590, lineHeight: "18px" }}>{title}</div>
      <div style={{ fontSize: 11, lineHeight: "14px", color: theme.text.tertiary, marginTop: 2 }}>{hint}</div>
    </button>
  );
}

function FlowRight() {
  const theme = useHostTheme();
  return (
    <span style={{ color: theme.text.quaternary, fontSize: 14, padding: "0 2px", alignSelf: "center" }}>
      →
    </span>
  );
}

function FlowDown() {
  const theme = useHostTheme();
  return (
    <div style={{ color: theme.text.quaternary, fontSize: 14, padding: "2px 0 2px 18px" }}>↓</div>
  );
}

function FlowRow({ children }: { children?: Parameters<typeof Row>[0]["children"] }) {
  return (
    <Row gap={6} align="center" wrap>
      {children}
    </Row>
  );
}

function FileRow({ files }: { files: FileRef[] }) {
  const dispatch = useCanvasAction();
  return (
    <Row gap={8} wrap align="center">
      {openRefs(dispatch, files)}
    </Row>
  );
}

function BlockDetail({ block }: { block: Block }) {
  return (
    <Card>
      <CardHeader trailing={<Text size="small">{block.hint}</Text>}>{block.title}</CardHeader>
      <CardBody>
        <Stack gap={10}>
          <Text size="small" tone="tertiary">
            {block.lane} · {block.laneNote}
          </Text>
          <H3>Role</H3>
          <Text>{block.role}</Text>
          <H3>Mechanism</H3>
          <Text>{block.mechanism}</Text>
          <H3>Built</H3>
          <Text>{block.built}</Text>
          <H3>Wired</H3>
          <Text>{block.wired}</Text>
          <Callout tone="warning" title="Invariant">
            {block.invariant}
          </Callout>
          <FileRow files={block.files} />
        </Stack>
      </CardBody>
    </Card>
  );
}

function SystemMap() {
  const theme = useHostTheme();
  const [blockId, setBlockId] = useCanvasState("mapBlock", "unit");
  const block = BLOCK_BY_ID[blockId] ?? BLOCKS[3];

  return (
    <Stack gap={16}>
      <div>
        <H2>System map</H2>
        <Text tone="secondary" size="small" style={{ marginTop: 4 }}>
          Click a block. Each one is a constructed object, not a folder name.
        </Text>
      </div>
      <Stack gap={14}>
        {LANES.map((lane, index) => (
          <div key={lane.id}>
            {index > 0 ? <FlowDown /> : null}
            <Text size="small" weight="semibold">
              {lane.id}
            </Text>
            <Text size="small" tone="tertiary">
              {lane.note}
            </Text>
            <div style={{ marginTop: 8 }}>
              <FlowRow>
                {lane.blocks.map((id, i) => {
                  const b = BLOCK_BY_ID[id];
                  return (
                    <span key={id} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                      {i > 0 ? <FlowRight /> : null}
                      <DiagramNode
                        title={b.title}
                        hint={b.hint}
                        active={blockId === id}
                        onClick={() => setBlockId(id)}
                      />
                    </span>
                  );
                })}
              </FlowRow>
            </div>
          </div>
        ))}
      </Stack>
      <Divider />
      <Grid columns="minmax(0, 1.4fr) minmax(240px, 0.8fr)" gap={16} align="start">
        <BlockDetail block={block} />
        <Stack gap={12}>
          <H3>Product boundary</H3>
          <Text size="small">
            User-visible: one Realtime assistant at <Code>/v1/realtime</Code>.
          </Text>
          <Text size="small">
            Internal: a pool of queue-driven cascades, one FastAPI process, swappable STT/LLM/TTS adapters.
          </Text>
          <Text size="small">
            Must stay live: VAD, the send loop, and listen. Generation is cancellable; the unit is not.
          </Text>
          <Text size="small" style={{ color: theme.text.secondary }}>
            Hidden from the client: pipeline index, session_id internals, CancelScope.generation, turn_revision, SESSION_END.
          </Text>
        </Stack>
      </Grid>
    </Stack>
  );
}

function RequestPath() {
  const [scenarioId, setScenarioId] = useCanvasState("pathScenario", "spoken");
  const [step, setStep] = useCanvasState("pathStep", 0);
  const scenario = SCENARIOS.find((s) => s.id === scenarioId) ?? SCENARIOS[0];
  const safeStep = Math.min(step, scenario.steps.length - 1);
  const current = scenario.steps[safeStep];
  const live = new Set(current.live);

  return (
    <Stack gap={16}>
      <div>
        <H2>Request path</H2>
        <Text tone="secondary" size="small" style={{ marginTop: 4 }}>
          Step a scenario. Live blocks are the objects that actually run on that beat.
        </Text>
      </div>
      <Row gap={8} wrap>
        {SCENARIOS.map((s) => (
          <span key={s.id}>
            <Pill
              active={s.id === scenario.id}
              onClick={() => {
                setScenarioId(s.id);
                setStep(0);
              }}
            >
              {s.title}
            </Pill>
          </span>
        ))}
      </Row>
      <Text size="small" tone="tertiary">
        {scenario.hint}
      </Text>
      <Row gap={8} wrap>
        {scenario.steps.map((s, i) => (
          <span key={s.hint}>
            <Pill active={i === safeStep} onClick={() => setStep(i)}>
              {i + 1}. {s.title}
            </Pill>
          </span>
        ))}
      </Row>
      <Stack gap={12}>
        {LANES.map((lane) => (
          <div key={lane.id}>
            <Text size="small" tone="tertiary">
              {lane.id}
            </Text>
            <div style={{ marginTop: 6 }}>
              <FlowRow>
                {lane.blocks.map((id, i) => {
                  const b = BLOCK_BY_ID[id];
                  return (
                    <span key={id} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                      {i > 0 ? <FlowRight /> : null}
                      <DiagramNode title={b.title} hint={b.hint} active={live.has(id)} dim={!live.has(id)} />
                    </span>
                  );
                })}
              </FlowRow>
            </div>
          </div>
        ))}
      </Stack>
      <Card>
        <CardHeader trailing={<Text size="small">{current.hint}</Text>}>{current.title}</CardHeader>
        <CardBody>
          <Stack gap={10}>
            <Text>{current.what}</Text>
            <Text size="small" tone="secondary">
              Wire: {current.wire}
            </Text>
            <FileRow
              files={current.live
                .map((id) => BLOCK_BY_ID[id].files[0])
                .filter((file, i, arr) => arr.findIndex((f) => f.path === file.path) === i)}
            />
          </Stack>
        </CardBody>
      </Card>
      <Row justify="space-between">
        <Button variant="ghost" disabled={safeStep === 0} onClick={() => setStep(Math.max(0, safeStep - 1))}>
          Previous
        </Button>
        <Text size="small" tone="quaternary">
          {safeStep + 1} / {scenario.steps.length}
        </Text>
        <Button
          variant={safeStep === scenario.steps.length - 1 ? "secondary" : "primary"}
          disabled={safeStep === scenario.steps.length - 1}
          onClick={() => setStep(Math.min(scenario.steps.length - 1, safeStep + 1))}
        >
          Next
        </Button>
      </Row>
    </Stack>
  );
}

function MachineView() {
  const [kind, setKind] = useCanvasState<MachineKind>("machineKind", "protocol");
  const [stateId, setStateId] = useCanvasState("machineState", "ready");
  const states = kind === "protocol" ? PROTOCOL_STATES : UNIT_STATES;
  const selected = states.find((s) => s.id === stateId) ?? states[0];

  return (
    <Stack gap={16}>
      <div>
        <H2>Work / protocol machine</H2>
        <Text tone="secondary" size="small" style={{ marginTop: 4 }}>
          Friendly name on the box. Real event or field on the hint line. Click a state for the transition that builds it.
        </Text>
      </div>
      <Row gap={8}>
        <span>
          <Pill
            active={kind === "protocol"}
            onClick={() => {
              setKind("protocol");
              setStateId("ready");
            }}
          >
            OpenAI Realtime events
          </Pill>
        </span>
        <span>
          <Pill
            active={kind === "unit"}
            onClick={() => {
              setKind("unit");
              setStateId("idle");
            }}
          >
            Unit + CancelScope
          </Pill>
        </span>
      </Row>
      <FlowRow>
        {states.map((s, i) => (
          <span key={s.id} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            {i > 0 ? <FlowRight /> : null}
            <DiagramNode
              title={s.title}
              hint={s.hint}
              active={selected.id === s.id}
              onClick={() => setStateId(s.id)}
            />
          </span>
        ))}
      </FlowRow>
      <Card>
        <CardHeader trailing={<Text size="small">{selected.hint}</Text>}>{selected.title}</CardHeader>
        <CardBody>
          <Stack gap={10}>
            <H3>Enters</H3>
            <Text>{selected.enters}</Text>
            <H3>Leaves</H3>
            <Text>{selected.leaves}</Text>
            <H3>Transition code</H3>
            <Text size="small">
              <Code>{selected.code}</Code>
            </Text>
            <FileRow files={selected.files} />
          </Stack>
        </CardBody>
      </Card>
      {kind === "protocol" ? (
        <Callout tone="info" title="Generate rule">
          conversation.item.create does not generate. response.create does. Interruption cancels the current spoken
          generation via CancelScope; it does not drop the claimed unit or user-side transcription events.
        </Callout>
      ) : (
        <Callout tone="warning" title="Release rule">
          Clearing session before SESSION_END drains would let the next client inherit the previous turn's transcript.
          Quarantine is safer than reuse.
        </Callout>
      )}
    </Stack>
  );
}

function BootWiring() {
  const [n, setN] = useCanvasState("bootStep", 1);
  const step = BOOT.find((s) => s.n === n) ?? BOOT[0];

  return (
    <Stack gap={16}>
      <div>
        <H2>Boot wiring</H2>
        <Text tone="secondary" size="small" style={{ marginTop: 4 }}>
          The actual new order when speech-to-speech --mode realtime starts. Not a dependency graph — a constructor sequence.
        </Text>
      </div>
      <FlowRow>
        {BOOT.map((s, i) => (
          <span key={s.n} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            {i > 0 ? <FlowRight /> : null}
            <DiagramNode
              title={`${s.n}. ${s.title}`}
              hint={s.hint}
              active={s.n === step.n}
              onClick={() => setN(s.n)}
            />
          </span>
        ))}
      </FlowRow>
      <Card>
        <CardHeader trailing={<Text size="small">{step.hint}</Text>}>
          {`${step.n}. ${step.title}`}
        </CardHeader>
        <CardBody>
          <Stack gap={10}>
            <H3>Constructor</H3>
            <Text>{step.ctor}</Text>
            <H3>Who news it</H3>
            <Text>{step.who}</Text>
            <H3>Must not import / do</H3>
            <Text>{step.mustNot}</Text>
            <FileRow files={step.files} />
          </Stack>
        </CardBody>
      </Card>
      <Row justify="space-between">
        <Button variant="ghost" disabled={step.n === 1} onClick={() => setN(Math.max(1, step.n - 1))}>
          Previous
        </Button>
        <Text size="small" tone="quaternary">
          {step.n} / {BOOT.length}
        </Text>
        <Button
          variant={step.n === BOOT.length ? "secondary" : "primary"}
          disabled={step.n === BOOT.length}
          onClick={() => setN(Math.min(BOOT.length, step.n + 1))}
        >
          Next
        </Button>
      </Row>
    </Stack>
  );
}

export default function SpeechToSpeechMechanism() {
  const [view, setView] = useCanvasState<ViewId>("view", "map");

  return (
    <Stack gap={24} style={{ maxWidth: 1180, margin: "0 auto", padding: "24px 28px 40px" }}>
      <Grid columns="minmax(0, 1.7fr) minmax(220px, 0.7fr)" gap={20} align="start">
        <div>
          <Text size="small" tone="tertiary" weight="semibold">
            SPEECH-TO-SPEECH · HOW IT IS BUILT
          </Text>
          <H1 style={{ marginTop: 6 }}>A cascade that pretends to be one voice</H1>
          <Text tone="secondary" style={{ marginTop: 8, maxWidth: 720 }}>
            OpenAI Realtime on the wire. Inside: one uvicorn, a pool of PipelineUnits, and six handler threads per
            unit — VAD, STT, notifier, LLM, tool split, TTS — joined by queues. Click a block to see who constructs it.
          </Text>
        </div>
        <Row gap={16} wrap>
          <Stat value="6" label="Handlers per unit" />
          <Stat value="16 kHz" label="Internal PCM" />
        </Row>
      </Grid>

      <Row gap={8} wrap>
        {VIEWS.map((v) => (
          <span key={v.id}>
            <Pill active={view === v.id} onClick={() => setView(v.id)}>
              {v.label}
            </Pill>
          </span>
        ))}
      </Row>

      {view === "map" ? <SystemMap /> : null}
      {view === "path" ? <RequestPath /> : null}
      {view === "machine" ? <MachineView /> : null}
      {view === "boot" ? <BootWiring /> : null}
    </Stack>
  );
}
